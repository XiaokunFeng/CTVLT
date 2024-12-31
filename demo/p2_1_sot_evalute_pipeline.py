## 利用 msma 跟跟踪结果进行评估
import os
import sys

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import gc
import resource
import argparse
import cv2
import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms

# import masa
from masa.apis import inference_masa, inference_masa_sot,init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry
from utils import filter_and_update_tracks
from utils_bbox import box_iou

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass


class DS_infor:
    def __init__(self, ds_name = "tnl2k"):
        ds_infor = {}
        if ds_name == "tnl2k":
            self.ds_save_dir = "/home/data_2/other_python_proj/masa/resource/eval_infor/tnl2k_res"
            if os.path.isdir(self.ds_save_dir) == False:
                os.mkdir(self.ds_save_dir)

            self.gt_data_path = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/TNL2K_test_subset'

            self.tracker_res_dir = "/home/data_2/other_python_proj/groundingLMM/resource/track_res/mempro/tnl2k"

            self.file2_folder = self.gt_data_path
            self.file_name_list = os.listdir(self.gt_data_path)
            self.file_name_list.sort()
        elif ds_name == "lasot":
            ds_save_dir = "/home/data_2/model_checkpoint_save/llm_track_res/lasot_first_frame_ref_res"
            if os.path.isdir(ds_save_dir) == False:
                os.mkdir(ds_save_dir)

            gt_data_path = '/home/data/wargame/fengxiaokun/LaSOT/data'
            file_dir_list = os.listdir(gt_data_path)
            file_dir_list.sort()
            # file_name_list = file_name_list[:2]
            for filedir in tqdm(file_dir_list):
                if filedir.endswith(".txt"):
                    continue
                file_path_list = os.listdir(os.path.join(gt_data_path, filedir))
                for file_name in file_path_list:
                    # gt路径
                    file_item = {}
                    nlp_path = os.path.join(gt_data_path, filedir, file_name,
                                            "nlp.txt")
                    img_dir = os.path.join(gt_data_path, filedir, file_name,
                                            "img")
                    img_list = os.listdir(img_dir)
                    img_list.sort()

                    first_frame_path = os.path.join(img_dir, img_list[0])
                    with open(nlp_path, 'r') as file:
                        # 读取文件内容
                        content = file.read()
                    if len(content) > 1:
                        if content[-1] == ".":
                            content = content[:-1]

                    file_item["initial_text"] = content
                    file_item["first_frame_path"] = first_frame_path
                    ds_infor[file_name] = file_item

    def get_seq_infor(self,index):
        filename =  self.file_name_list[index]
        self.seq_name = filename.split(".")[0]
        # gt路径
        file_item = {}
        nlp_path = os.path.join(self.file2_folder, filename.split(".")[0],
                                "language.txt")
        img_dir = os.path.join(self.file2_folder, filename.split(".")[0],
                               "imgs")
        self.seq_img_dir = img_dir
        img_list = os.listdir(img_dir)
        img_list.sort()

        first_frame_path = os.path.join(self.file2_folder, filename.split(".")[0],
                                        "imgs", img_list[0])

        with open(nlp_path, 'r') as file:
            # 读取文件内容
            content = file.read()
        if len(content) > 1:
            if content[-1] == ".":
                content = content[:-1]

        file_item["initial_text"] = content
        file_item["first_frame_path"] = first_frame_path
        file_item["img_list"] = img_list

        gt_bbox_path = os.path.join(self.file2_folder, filename.split(".")[0],
                                    "groundtruth.txt")

        gt_bbox = np.loadtxt(gt_bbox_path, delimiter=",")
        file_item["gt_bbox"] = gt_bbox.tolist()

        tracker_res_dir = os.path.join(self.tracker_res_dir,filename.split(".")[0] + ".txt")

        tracker_res = np.loadtxt(tracker_res_dir, delimiter='\t')
        file_item["tracker_res"] = tracker_res.tolist()

        return file_item
def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

# Set the file descriptor limit to 65536
set_file_descriptor_limit(65536)

def visualize_frame(args, visualizer, frame, track_result, frame_idx, fps=None):
    visualizer.add_datasample(
        name='video_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=args.score_thr,
        fps=fps,)
    frame = visualizer.get_image()
    gc.collect()
    return frame

def parse_args():

    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('--video',default= "minions_rush_out.mp4", help='Video file')
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', default= "../configs/masa-gdino/masa_gdino_swinb_inference.py", help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint',default= "../resource/pretrained_model/gdino_masa.pth", help='Masa Checkpoint file')
    parser.add_argument( '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--out', type=str,default="../resource/outputs/", help='Output video file')
    parser.add_argument('--save_dir', type=str,default="../resource/outputs/", help='Output for video frames')
    # parser.add_argument('--texts', default= "yellow_minions",help='text prompt')
    parser.add_argument('--line_width', type=int, default=5, help='Line width')
    parser.add_argument('--unified', default= True, help='Use unified model, which means the masa adapter is built upon the detector model.')
    parser.add_argument('--detector_type', type=str, default='mmdet', help='Choose detector type')
    parser.add_argument('--fp16', action='store_true', help='Activation fp16 mode')
    parser.add_argument('--no-post', action='store_true', help='Do not post-process the results ')
    parser.add_argument('--show_fps',default= True, help='Visualize the fps')
    parser.add_argument('--sam_path',  type=str, default='saved_models/pretrain_weights/sam_vit_h_4b8939.pth', help='Default path for SAM models')
    parser.add_argument('--sam_type', type=str, default='vit_h', help='Default type for SAM models')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert args.out, \
        ('Please specify at least one operation (save the '
         'video) with the argument "--out" ')

    # build the model from a config file and a checkpoint file
    print("init masa model begin...")
    masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    print("init masa model done !!")


    #### parsing the text input
    texts = "place holder"
    if texts is not None:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
        masa_model.cfg.visualizer['texts'] = texts
    else:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg)
        # masa_model.cfg.visualizer['texts'] = det_model.dataset_meta['classes']

    # init visualizer
    masa_model.cfg.visualizer['save_dir'] = args.save_dir
    masa_model.cfg.visualizer['line_width'] = args.line_width

    # analyze tracker datasets
    track_ds =DS_infor()
    seq_num = len(track_ds.file_name_list)
    for seq_index in tqdm.tqdm(range(seq_num)):
        seq_item  = track_ds.get_seq_infor(seq_index)
        save_path = os.path.join(track_ds.ds_save_dir, "%s.pth" % (track_ds.seq_name))
        if os.path.isfile(save_path):
            continue

        frame_idx = 0
        embed_list = []
        bbox_infor = []   # pred_bbox; gt_bbox; iou;

        frame_list = seq_item["img_list"]
        pred_bbox = seq_item["tracker_res"]
        pred_bbox = torch.tensor(pred_bbox)
        pred_bbox[:,2] = pred_bbox[:,2] + pred_bbox[:,0]
        pred_bbox[:, 3] = pred_bbox[:, 3] + pred_bbox[:, 1]

        gt_bbox = seq_item["gt_bbox"]
        gt_bbox = torch.tensor(gt_bbox)
        gt_bbox[:, 2] = gt_bbox[:, 2] + gt_bbox[:, 0]
        gt_bbox[:, 3] = gt_bbox[:, 3] + gt_bbox[:, 1]

        iou,_ = box_iou(gt_bbox,pred_bbox)

        video_len = len(frame_list)
        fps_list = []
        for frame_index, frame_name in enumerate(tqdm.tqdm(frame_list)):
            frame = cv2.imread(os.path.join(track_ds.seq_img_dir,frame_name))
            # unified models mean that masa build upon and reuse the foundation model's backbone features for tracking
            track_result = inference_masa_sot(masa_model, frame,
                                          frame_id=frame_idx,
                                          video_len= video_len,  # len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          text_prompt=texts,
                                            bbox = pred_bbox[frame_index],
                                          fp16=args.fp16,
                                          detector_type=args.detector_type,
                                          show_fps=args.show_fps)
            if args.show_fps:
                track_result, fps = track_result
            frame_idx += 1

            # print(track_result,fps)
            embed_list.append(track_result)
            # frames.append(frame)
            fps_list.append(fps)
        # save_res

        embed_data = torch.cat(embed_list,dim=0)  # N, 256
        embed_data = torch.cat([embed_data,iou.unsqueeze(-1).to(embed_data.device)],dim=1)  # N, (256+1) ; emb,iou
        torch.save(embed_data,save_path)

        print(track_ds.seq_name," fps: ", np.mean(fps_list) )

    print('Done')


if __name__ == '__main__':
    main()

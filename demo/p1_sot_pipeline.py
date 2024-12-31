import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_root)

import gc
import resource
import argparse
import cv2
import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method

from mmdet.registry import VISUALIZERS

# import masa
from masa.apis import inference_masa, inference_masa_sot,init_masa, build_test_pipeline
from utils_demo import filter_and_update_tracks

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

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
    texts = "the gun"
    if texts is not None:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
        masa_model.cfg.visualizer['texts'] = texts
    else:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg)
        # masa_model.cfg.visualizer['texts'] = det_model.dataset_meta['classes']


    # init visualizer
    masa_model.cfg.visualizer['save_dir'] = args.save_dir
    masa_model.cfg.visualizer['line_width'] = args.line_width
    visualizer = VISUALIZERS.build(masa_model.cfg.visualizer)


    frame_idx = 0
    instances_list = []
    frames = []
    fps_list = []

    # img_dir = "/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/TNL2K_test_subset/advSamp_NBA2k_Kawayi_video_01-Done/imgs"
    img_dir = "/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/TNL2K_test_subset/test_039_COD_video_01_done/imgs"
    frame_list = os.listdir(img_dir)
    frame_list.sort()
    frame_list = frame_list[50:500]
    print(frame_list)
    video_len = len(frame_list)
    # out videowriter
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # first_frame = cv2.imread(os.path.join(img_dir,frame_list[0]))
    # video_writer = cv2.VideoWriter(
    #     args.out, fourcc, 30,
    #     (first_frame.shape[1], first_frame.shape[0]))

    for frame_name in tqdm.tqdm(frame_list):
        frame = cv2.imread(os.path.join(img_dir,frame_name))
        # unified models mean that masa build upon and reuse the foundation model's backbone features for tracking
        track_result = inference_masa(masa_model, frame,
                                      frame_id=frame_idx,
                                      video_len= video_len,  # len(video_reader),
                                      test_pipeline=masa_test_pipeline,
                                      text_prompt=texts,
                                      fp16=args.fp16,
                                      detector_type=args.detector_type,
                                      show_fps=args.show_fps)
        if args.show_fps:
            track_result, fps = track_result

        frame_idx += 1

        # print(track_result,fps)
        track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
        instances_list.append(track_result.to('cpu'))
        frames.append(frame)
        fps_list.append(fps)

    if not args.no_post:
        instances_list = filter_and_update_tracks(instances_list, (frame.shape[1], frame.shape[0]))

    # visualize results
    print('Start to visualize the results...')

    frames_with_res = []
    for idx, (frame, fps, track_result) in enumerate(zip(frames, fps_list, instances_list)):
        print(track_result)
        frame_res_item = visualize_frame(args, visualizer, frame, track_result.to('cpu'), idx, fps)

        # frames_with_res.append( frame_res_item )
        frame_res_item = cv2.cvtColor(frame_res_item, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(args.out,"frame_%d.png"%(idx) ),frame_res_item)

    print('Done')


if __name__ == '__main__':
    main()

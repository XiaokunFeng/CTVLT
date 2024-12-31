## 在 评估的基础上， 看一下 计算得到的 相似度大小， 效果如何
## 利用 msma 跟跟踪结果进行评估
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
import torch.nn.functional as F
import gc
import resource
import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method

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


def cal_match_scores(embeds,memo_embeds, tgt_memory_weight=None):
    embeds = torch.tensor(embeds).unsqueeze(0)
    memo_embeds = torch.stack(memo_embeds,dim=0)
    feats = torch.mm(embeds, memo_embeds.t())
    d2t_scores = feats.softmax(dim=1)
    t2d_scores = feats.softmax(dim=0)
    match_scores_bisoftmax = (d2t_scores + t2d_scores) / 2

    match_scores_cosine = torch.mm(
        F.normalize(embeds, p=2, dim=1),
        F.normalize(memo_embeds, p=2, dim=1).t(),
    )
    match_scores = (match_scores_bisoftmax + match_scores_cosine) / 2
    if tgt_memory_weight is None:
        match_scores_mean = torch.mean(match_scores)
    else:
        tgt_memory_weight = torch.stack(tgt_memory_weight,dim=0)
        tgt_memory_weight = tgt_memory_weight/torch.sum(tgt_memory_weight)
        match_scores_mean  = match_scores*tgt_memory_weight
        match_scores_mean = torch.sum(match_scores_mean)
    return match_scores,match_scores_mean

# Set the file descriptor limit to 65536
set_file_descriptor_limit(65536)


def main():
    # analyze tracker datasets
    track_ds =DS_infor()
    seq_num = len(track_ds.file_name_list)
    iou_all = []
    iou_pred_all = []
    # seq_num = 2
    for seq_index in tqdm.tqdm(range(seq_num)):
        seq_item  = track_ds.get_seq_infor(seq_index)
        res_path = os.path.join(track_ds.ds_save_dir, "%s.pth" % (track_ds.seq_name))
        embed_data = torch.load(res_path)   # N, (256+1) ; emb,iou
        iou_data = embed_data[:,-1]
        iou_all.append(iou_data)
        embed_data = embed_data[:,:-1]

        frame_list = seq_item["img_list"]
        pred_bbox = seq_item["tracker_res"]
        pred_bbox = torch.tensor(pred_bbox)
        pred_bbox[:,2] = pred_bbox[:,2] + pred_bbox[:,0]
        pred_bbox[:, 3] = pred_bbox[:, 3] + pred_bbox[:, 1]

        gt_bbox = seq_item["gt_bbox"]
        gt_bbox = torch.tensor(gt_bbox)
        gt_bbox[:, 2] = gt_bbox[:, 2] + gt_bbox[:, 0]
        gt_bbox[:, 3] = gt_bbox[:, 3] + gt_bbox[:, 1]

        sim_res = []
        tgt_memory = []
        tgt_memory_weight = []
        sim_res_every_time_infor = []
        memory_len = 20
        for frame_index, frame_name in enumerate(tqdm.tqdm(frame_list)):
            embed_item = embed_data[frame_index]

            ### way1
            # if len(tgt_memory) == 0:
            #     tgt_memory.append(embed_item)
            #     sim_res.append(torch.tensor(1).to(embed_item.device))
            # else:
            #     res,res_mean = cal_match_scores(embed_item,tgt_memory)
            #     sim_res.append(res_mean)
            #     if len(tgt_memory) < memory_len:
            #         tgt_memory.append(embed_item)
            #     else:
            #         tgt_memory.pop(0)
            #         tgt_memory.append(embed_item)

            # ### way2
            # if len(tgt_memory) == 0:
            #     tgt_memory.append(embed_item)
            #     sim_res.append(torch.tensor(1).to(embed_item.device))
            # else:
            #     res,res_mean = cal_match_scores(embed_item,tgt_memory)
            #     sim_res.append(res_mean)
            #     if res_mean < 0.5:
            #         continue
            #     else:
            #         if len(tgt_memory) < memory_len:
            #             tgt_memory.append(embed_item)
            #         else:
            #             tgt_memory.pop(0)
            #             tgt_memory.append(embed_item)

            ### way3  ema
            # if len(tgt_memory) == 0:
            #     tgt_memory.append(embed_item)
            #     sim_res.append(torch.tensor(1).to(embed_item.device))
            # else:
            #     res, res_mean = cal_match_scores(embed_item, tgt_memory)
            #     sim_res.append(res_mean)
            #     if res_mean < 0.5:
            #         continue
            #     else:
            #         if len(tgt_memory) < memory_len:
            #             tgt_memory.append(embed_item)
            #         else:
            #
            #             sim_res_every_time_infor.append(res)
            #             tgt_memory[1] = 0.2*tgt_memory[1] + 0.8*tgt_memory[2]
            #             tgt_memory[2:-1] = tgt_memory[3:]
            #             tgt_memory[-1] = embed_item

            ### way4
            if len(tgt_memory) == 0:
                tgt_memory.append(embed_item)
                tgt_memory_weight.append( torch.tensor(1).to(embed_item.device) )
                sim_res.append(torch.tensor(1).to(embed_item.device))
            else:
                if frame_index%5 != 0:
                    sim_res.append(sim_res[-1])
                    continue

                res, res_mean = cal_match_scores(embed_item, tgt_memory, tgt_memory_weight)
                sim_res.append(res_mean)


                if res_mean < 0.5:
                    continue
                else:
                    if len(tgt_memory) < memory_len:
                        tgt_memory.append(embed_item)
                        tgt_memory_weight.append(res_mean)
                    else:
                        tgt_memory[1] = 0.2*tgt_memory[1] + 0.8*tgt_memory[2]
                        tgt_memory[2:-1] = tgt_memory[3:]
                        tgt_memory[-1] = embed_item

                        tgt_memory_weight[1] = 0.2 * tgt_memory_weight[1] + 0.8 * tgt_memory_weight[2]
                        tgt_memory_weight[2:-1] = tgt_memory_weight[3:]
                        tgt_memory_weight[-1] = res_mean

        sim_res  = torch.stack(sim_res,dim=0)
        iou_pred_all.append(sim_res)

        sim_res = sim_res.cpu().numpy()

        plt.clf()
        plt.plot(iou_data.cpu().numpy())
        plt.plot(sim_res)
        plt.title(track_ds.seq_name)
        # plt.show()
        plt.savefig( os.path.join(track_ds.ds_save_dir,"imgs", track_ds.seq_name+".jpg" ))

    # cal res
    iou_pred_all = torch.cat(iou_pred_all,dim=0)
    iou_all = torch.cat(iou_all,dim=0)
    l1_loss = torch.mean(torch.abs(iou_all - iou_pred_all))
    l2_loss = torch.mean((iou_all - iou_pred_all)**2)
    print('Done',l1_loss,l2_loss )


if __name__ == '__main__':
    main()

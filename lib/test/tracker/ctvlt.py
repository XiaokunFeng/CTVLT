import math
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt

from lib.models.ctvlt import build_ctvlt
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target,sample_target_for_cam
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.utils.ce_utils import generate_mask_cond
import pandas as pd


# masa
import gc
import resource
import argparse
import cv2
import tqdm

import torch
from lib.test.evaluation.environment import env_settings

# from mmdet.registry import VISUALIZERS

# import masa
from masa.apis import inference_masa, inference_masa_sot,init_masa, build_test_pipeline
from demo.utils_demo import filter_and_update_tracks
import torchvision.transforms as transforms

def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
def visualize_grid_attention_v2(img, attention_mask, ratio=1, cmap="jet", save_image=True,
                                save_path="test.jpg", quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    # print("load image from: ", img_path)
    # img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.clf()
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=quality)
    #

class CTVLT(BaseTracker):
    def __init__(self, params):
        super(CTVLT, self).__init__(params)
        # load tracker
        network = build_ctvlt(params.cfg,training=False)
        missing_keys, unexpected_keys =network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        print("load tracker, checkpoint:", self.params.checkpoint)

        # load masa
        print("loading MASA...")
        # masa
        self.re_detec_by_masa =   False
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.masa_model = init_masa(
            os.path.join(self.current_dir,"../../../configs/masa-gdino/masa_gdino_swinb_inference.py"),
            os.path.join(self.current_dir, "../../../resource/pretrained_networks/gdino_masa.pth"),
            device='cuda')

        self.masa_test_pipeline = build_test_pipeline(self.masa_model.cfg, with_text=True)
        # self.masa_actor_pipeline = None
        self.resize_transform = transforms.Resize((24, 24))
        print("loading MASA done! ...")

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None


        # add for mgit
        env = env_settings()
        self.mgit_lable_dir = env.mgit_lable_dir
        try:
            self.dataset_name = params.dataset
        except:
            self.dataset_name = ""

        if "videocube" in self.dataset_name:
            self.action_level = 1
            self.activity_level = 0
            self.story_level = 0
            print(self.dataset_name)

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # temporal_len
        self.temporal_memory_len = 4
        self.memory_embed_len = 10
        self.memory_embed = []
        self.memory_embed_weight = []
        self.memory_embed_sim_res = []
        self.memory_embed_save = []
        self.memory_embed_neg_list = []   # 连续 n 次，低于给定阈值，就采用重定位机制

        # only by grounding
        self.only_grounding_flag = False
        self.grounding_res_bbox = {}

        self.s_vl_item = None
        self.vl_trans_intervel = env.vl_trans_intervel

    def initialize(self, image, info: dict,rope_initialize_flag=False):
        if "seq_frames_num" in info:
            self.video_len = info["seq_frames_num"]
        self.frame_index = 0
        # add for MGIT
        if 'videocube' in self.dataset_name:
            action_level = self.action_level
            activity_level = self.activity_level
            story_level = self.story_level
            self.frame_index = 0
            self.actions = []
            self.activities = []
            self.story = []
            self.action_start_frames = []
            self.action_end_frames = []
            self.activity_start_frames = []
            self.activity_end_frames = []
            self.story_start_frames = []
            self.story_end_frames = []

            self.grounding_res_bbox = {}

            seq_name = info["seq_name"]
            print(seq_name)
            dataset_tab_path = os.path.join(self.mgit_lable_dir, seq_name + '.xlsx')
            dataset_tab = pd.read_excel(dataset_tab_path, index_col=0)
            tab_activity = dataset_tab['activity': 'activity']
            tab_action = dataset_tab['action': 'action']
            tab_story = dataset_tab['story': 'story']
            for index, row in tab_action.iterrows():
                self.action_start_frames.append(row['start_frame'])
                self.action_end_frames.append(row['end_frame'])
                self.actions.append(row['description'])
            for index, row in tab_activity.iterrows():
                self.activity_start_frames.append(row['start_frame'])
                self.activity_end_frames.append(row['end_frame'])
                self.activities.append(row['description'])
            for index, row in tab_story.iterrows():
                self.story_start_frames.append(row['start_frame'])
                self.story_end_frames.append(row['end_frame'])
                self.story.append(row['description'])

            if action_level:
                info['init_nlp'] = self.actions[0]
                print('language', info['init_nlp'])
            elif activity_level:
                info['init_nlp'] = self.activities[0]
                print('language', info['init_nlp'])
            elif story_level:
                info['init_nlp'] = self.story[0]
                print('language', info['init_nlp'])

            info['init_text_description'] = info['init_nlp']

        try:
            z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                    output_sz=self.params.template_size)
        except:
            return  False
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.device = template.tensors.device

        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # Run Language Network
        self.text_description = info['init_text_description']


        self.memory_embed = []
        self.memory_embed_weight = []
        self.memory_embed_sim_res =[]
        self.memory_embed_reatrack_flag = []
        self.memory_embed_save = []

        # memoey embed save dir
        self.seq_name = info["seq_name"]
        # save states
        self.state = info['init_bbox']
        self.text_description = info['init_text_description']
        self.frame_id = 0
        self.tgt_all = []
        self.tgt_text_all = []
        self.tgt_decode_object_pre_all = []
        self.tgt_decode_context_pre_all = []

        self.s_vl_item = None
        if rope_initialize_flag:
            return True
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}


    def cal_match_scores(self, embeds,memo_embeds, tgt_memory_weight=None ):
        embeds = torch.tensor(embeds).view(1,256)
        memo_embeds = torch.stack(memo_embeds, dim=0).view(-1,256)
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
            tgt_memory_weight = torch.stack(tgt_memory_weight, dim=0)
            tgt_memory_weight = tgt_memory_weight / torch.sum(tgt_memory_weight)
            match_scores_mean = match_scores * tgt_memory_weight
            match_scores_mean = torch.sum(match_scores_mean)
        return match_scores, match_scores_mean

    def track(self, image, info: dict = None):
        if self.only_grounding_flag == False:
            return  self.track_by_joint(image,info)
        else:
            return  self.only_ground(image, info)
    def only_ground(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        masa_track_res = inference_masa(self.masa_model, image,
                                              frame_id= 0,    # reset the memory
                                              video_len=self.video_len,  # len(video_reader),
                                              test_pipeline=self.masa_test_pipeline,
                                              text_prompt=self.text_description,
                                              fp16=False,
                                              detector_type='mmdet',
                                              show_fps=False)
        masa_track_res_infor = masa_track_res[0].pred_track_instances
        # masa_track_res_scores = masa_track_res_infor.scores

        self.grounding_res_bbox[self.frame_id] = masa_track_res_infor


        return None

    def track_by_joint(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # add for MGIT
        if 'videocube' in self.dataset_name:
            activity_level = self.activity_level
            action_level = self.action_level
            story_level = self.story_level

            if action_level:
                action_start_frames = self.action_start_frames
                action_end_frames = self.action_end_frames
                actions = self.actions
                for i in range(0, len(action_start_frames)):
                    if self.frame_id >= action_start_frames[i] and self.frame_id <= action_end_frames[i]:
                        if self.frame_index != i:
                            self.frame_index += 1
                            print('action_level self.frame_index', self.frame_index)
                            print('actions', actions[i])
                            self.text_description = actions[i]
                        break
                    else:
                        continue
            elif activity_level:
                activity_start_frames = self.activity_start_frames
                activity_end_frames = self.activity_end_frames
                activities = self.activities
                for i in range(0, len(activity_start_frames)):
                    if self.frame_id >= activity_start_frames[i] and self.frame_id <= activity_end_frames[i]:
                        if self.frame_index != i:
                            self.frame_index += 1
                            print('activity_level self.frame_index', self.frame_index)
                            print('activities', activities[i])
                        break
                    else:
                        continue
            elif story_level:
                story_start_frames = self.story_start_frames
                story_end_frames = self.story_end_frames
                story = self.story
                for i in range(0, len(story_start_frames)):
                    if self.frame_id >= story_start_frames[i] and self.frame_id <= story_end_frames[i]:
                        if self.frame_index != i:
                            self.frame_index += 1
                            print('story_level self.frame_index', self.frame_index)
                            print('story', story[i])
                            self.text_features, self.text_sentence_features = self.network.forward_text(
                                [story[i]], num_search=1, device=self.device)
                        break
                    else:
                        continue

        x_patch_arr, resize_factor, x_amask_arr, crop_infor, pad_infor = sample_target_for_cam(image, self.state,
                                                                                               self.params.search_factor,
                                                                                               output_sz=self.params.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # mappding the nl to heatmap by grounding dino
        search_tensor = search.tensors[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(search_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(search_tensor.device)
        search_np = (search_tensor* std + mean)*255
        search_np = search_np.permute(1, 2, 0).cpu().numpy()
        if self.frame_id % self.vl_trans_intervel == 0  or self.s_vl_item is None:
            self.s_vl_item = inference_masa_sot(self.masa_model, search_np,
                                           frame_id=0,  # reset the memory
                                           video_len=10,  # len(video_reader),
                                           test_pipeline=self.masa_test_pipeline,
                                           text_prompt=self.text_description,
                                           fp16=False,
                                           detector_type='mmdet',
                                           show_fps=False,
                                           mode="mapping")                      # 100,100

        with torch.no_grad():
            x_dict = search
            # run visual encoder and decoder
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=x_dict.tensors,
                exp_str_trans=self.s_vl_item,
                training=False, tgt_pre=self.tgt_all,
            )



        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return CTVLT

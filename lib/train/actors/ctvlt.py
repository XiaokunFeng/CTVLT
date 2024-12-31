import copy
import os

import cv2
import numpy as np
from PIL import Image
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy,box_iou
import torch
from torch import nn
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

import numpy
from lib.apis.acc_eval import accuracy
from lib.train.admin import multigpu
import pycocotools.mask as maskUtils

import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size

# import masa
from masa.apis import inference_masa, inference_masa_sot,init_masa, build_test_pipeline
from demo.utils_demo import filter_and_update_tracks


class CTVLTActor(BaseActor):
    """ Actor for training VLTrack models """

    def __init__(self, net, objective, settings, loss_weight=None, cfg=None,
                det_coord=[0], det_coord_weight=1.5):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.det_coord = det_coord
        self.det_coord_weight = det_coord_weight

        ## paras for loss back
        self.setting = None
        self.use_amp = None
        self.optimizer = None
        self.scaler = GradScaler()

        # reg loss
        self.confidence_reg_loss = nn.MSELoss()

        # sub mask index pred loss
        self.sub_mask_index_cls_loss = nn.BCELoss()

        self.output_nan_flag = False
        self.track_the_bbox = False

        # load masa
        print("loading MASA...")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        self.masa_model = init_masa(
            os.path.join(self.current_dir,"../../../configs/masa-gdino/masa_gdino_swinb_inference.py"),
            os.path.join(self.current_dir, "../../../resource/pretrained_networks/gdino_masa.pth"),
            device='cuda:0')
        self.masa_actor_pipeline = build_test_pipeline(self.masa_model.cfg, with_text=True)
        self.resize_transform = transforms.Resize((24, 24))
        print("loading MASA done! ...")


    def __call__(self, data):

        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status
    def forward_pass(self, data):

        ############ map the sentense to heatmap_data  ##########################
        exp_str_subject_mask_infor = data["exp_str"]
        exp_str_list = []
        subject_mask_list = []
        for item in exp_str_subject_mask_infor:
            item_list = item.split("+")
            exp_str_list.append(item_list[0])
            index_list = list(map(int, item_list[-1].split(",")))
            subject_mask_list.append(index_list)

        search_images = data["search_images"]
        temporal_len, batch_size = search_images.shape[0], search_images.shape[1]
        trans_infor = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(search_images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(search_images.device)

        for batch_index in range(batch_size):
            batch_item = search_images[:,batch_index]
            exp_item = exp_str_list[batch_index]
            for frame_index in range(temporal_len):
                frame_item  =  batch_item[frame_index]
                # back to np.arr
                frame_item = (frame_item* std + mean)*255
                frame_item = frame_item.permute(1, 2, 0).cpu().numpy()

                # ### get the tracking res
                if self.track_the_bbox:
                    masa_track_item = inference_masa(self.masa_model, frame_item,
                                                    frame_id=0,  # reset the memory
                                                    video_len=temporal_len,  # len(video_reader),
                                                    test_pipeline=self.masa_actor_pipeline,
                                                    text_prompt=exp_item,
                                                    fp16=False,
                                                    detector_type='mmdet',
                                                    show_fps=False)
                    # masa_track_res_infor = masa_track_item[0].pred_track_instances.bboxes


                ## get the mapping query
                s_vl_item = inference_masa_sot(self.masa_model, frame_item,
                                                 frame_id=0,  # reset the memory
                                                 video_len=temporal_len,  # len(video_reader),
                                                 test_pipeline=self.masa_actor_pipeline,
                                                 text_prompt=exp_item,
                                                 fp16=False,
                                                 detector_type='mmdet',
                                                 show_fps=False,
                                                 mode="mapping")

                trans_infor.append(s_vl_item)



        template_list, search_list = [], []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img_i)


        out_dict = self.net(template=template_list,
                            search=search_list,
                            exp_str_trans = trans_infor,
                            return_last_attn=False)


        return out_dict

    def compute_losses(self,pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        #gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_bbox = gt_dict['search_anno'].view(-1,4)
        gts = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(gts, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss  # + confidence_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()
                    }
            return loss, status
        else:
            return loss

    def calc_accuracy(self, out_dict, bbox_anno=None, mask_anno=None,search_index=0):
        det_acc_list, mask_iou_list, mask_acc_list = [], [], []
        
        gt_bbox = box_xywh_to_xyxy(bbox_anno)[search_index]  # norm coords: (x1,y1,w,h) to (X1, Y1, X2, Y2)
        with torch.no_grad():
            batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                pred_bboxes=out_dict['pred_bboxes'] * self.settings.output_sz['search'],
                gt_bbox=gt_bbox * self.settings.output_sz['search'],
                device=out_dict['pred_bboxes'].device)

        det_acc_list.append(batch_det_acc.item())
        bbox_acc = sum(det_acc_list) / len(det_acc_list)



        return bbox_acc, None, None


    def calc_accuracy_and_iou(self, out_dict, bbox_anno=None, mask_anno=None,search_index=0):
        det_acc_list, mask_iou_list, mask_acc_list = [], [], []

        gt_bbox = box_xywh_to_xyxy(bbox_anno)[search_index]  # norm coords: (x1,y1,w,h) to (X1, Y1, X2, Y2)

        with torch.no_grad():
            batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                pred_bboxes=out_dict['pred_bboxes'] * self.settings.output_sz['search'],
                gt_bbox=gt_bbox * self.settings.output_sz['search'],
                device=out_dict['pred_bboxes'].device)

            iou, _ = box_iou(out_dict['pred_bboxes'] * self.settings.output_sz['search'],
                             gt_bbox * self.settings.output_sz['search'])

        det_acc_list.append(batch_det_acc.item())
        bbox_acc = sum(det_acc_list) / len(det_acc_list)

        return bbox_acc, iou.mean()

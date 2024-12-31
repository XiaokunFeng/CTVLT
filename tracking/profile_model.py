import os
import sys

import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib



# import masa
from masa.apis import inference_masa, inference_masa_sot,init_masa, build_test_pipeline
from demo.utils_demo import filter_and_update_tracks
import torchvision.transforms as transforms


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ctvlt', choices=['ctvlt'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()

    return args

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.join(current_dir,"../")
masa_model = init_masa(
            os.path.join(proj_dir,"/configs/masa-gdino/masa_gdino_swinb_inference.py"),
            os.path.join(proj_dir,"/resource/pretrained_networks/gdino_masa.pth"),
            device='cuda')
masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
# self.masa_actor_pipeline = None
resize_transform = transforms.Resize((24, 24))
print("loading MASA done! ...")


def evaluate_vit(model, template, search,text_features):
    '''Speed Test'''
    training = False
    force_text_forward = True
    search_np = np.ones((224,224,3))
    text_description = "This is for speed "
    s_vl_item = inference_masa_sot(masa_model, search_np,
                                        frame_id=0,  # reset the memory
                                        video_len=10,  # len(video_reader),
                                        test_pipeline=masa_test_pipeline,
                                        text_prompt=text_description,
                                        fp16=False,
                                        detector_type='mmdet',
                                        show_fps=False,
                                        mode="mapping")  # 100,100
    print("s_vl_item' shape: ",s_vl_item.shape)

    macs1, params1 = profile(model, inputs=(template, search,s_vl_item, False, False,[]
                                            ),
                             custom_ops=None, verbose=False)

    print("training: ",training)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 2000
    d_t = 5
    print("testing speed ... ",d_t)
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, search,s_vl_item, False, False,[])
        start = time.time()
        for i in range(T_t):
            if i% d_t == 0:
                s_vl_item = inference_masa_sot(masa_model, search_np,
                                               frame_id=0,  # reset the memory
                                               video_len=10,  # len(video_reader),
                                               test_pipeline=masa_test_pipeline,
                                               text_prompt=text_description,
                                               fp16=False,
                                               detector_type='mmdet',
                                               show_fps=False,
                                               mode="mapping")  # 100,100
            _ = model(template, search,s_vl_item, False, False,[])
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))


def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = '../experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    # if args.script == "ostrack":
    model_module = importlib.import_module('lib.models')
    model_constructor = model_module.build_ctvlt
    model = model_constructor(cfg, training=False)
    # get the template and search
    template = torch.randn(bs, 3, z_sz, z_sz)
    search = torch.randn(bs, 3, x_sz, x_sz)
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    # text_features = NestedTensor(torch.ones(1, 6,512).to(device), torch.ones(1, 6).bool().to(device))
    text_features = ["This is for speed "]
    merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
    # if merge_layer <= 0:
    for _ in range(10):
        evaluate_vit(model, template, search, text_features)
    # else:
    #     evaluate_vit_separate(model, template, search)

    # else:
    #     raise NotImplementedError

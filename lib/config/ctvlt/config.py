from easydict import EasyDict as edict
import yaml

"""
Add default config for Seqtrack_segmentation.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.PRETRAIN_CHECKPOINT = ""
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.re_detec_by_masa = True

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.TRANSFORMER_DEC
cfg.MODEL.TRANSFORMER_DEC = edict()
cfg.MODEL.TRANSFORMER_DEC.NHEADS = 8
cfg.MODEL.TRANSFORMER_DEC.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER_DEC.DIM_FEEDFORWARD = 512
cfg.MODEL.TRANSFORMER_DEC.ENC_LAYERS = 0
cfg.MODEL.TRANSFORMER_DEC.DEC_LAYERS = 3
cfg.MODEL.TRANSFORMER_DEC.PRE_NORM = False
cfg.MODEL.TRANSFORMER_DEC.DIVIDE_NORM = False

#position encoding
cfg.MODEL.HIDDEN_DIM = 512
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = 'learned'

# TEXT_ENCODER
cfg.MODEL.TEXT_ENCODER = 'roberta-base'
cfg.MODEL.FREEZE_TEXT_ENCODER = True

# VISION LANGUAGE ENCODER
cfg.MODEL.VLFUSION_LAYERS = 0
cfg.MODEL.VL_INPUT_TYPE = 'separate'
cfg.MODEL.CLS_TOKEN_LEN = 1
cfg.MODEL.LEN_TEMPORAL = 4

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.DEC_LAYERS = 0
cfg.MODEL.DECODER.HIDDEN_DIM = 256
cfg.MODEL.DECODER.MLP_RATIO = 8
cfg.MODEL.DECODER.NUM_HEADS = 8
cfg.MODEL.DECODER.DROPOUT = 0.1
cfg.MODEL.DECODER.VOCAB_SIZE = 1001
cfg.MODEL.DECODER.BBOX_TYPE = 'xyxy'
cfg.MODEL.DECODER.MEMORY_POSITION_EMBEDDING = "sine"
cfg.MODEL.DECODER.QUERY_POSITION_EMBEDDING = "learned"

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.BBOX_TASK = False
cfg.TRAIN.LANGUAGE_TASK = False
cfg.TRAIN.AUX_LOSS = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone
cfg.TRAIN.USE_TEMPORAL_INFOR = True
cfg.TRAIN.TEMPORAL_ENCODER_FLAG = False
cfg.TRAIN.ONLY_TRAIN_BACK = False

cfg.TRAIN.TRAIN_PROMPT_MODULE = True
cfg.TRAIN.TRAIN_IMG_PROMPT_MODULE = True
cfg.TRAIN.TRAIN_TEXT_PROMPT_MODULE = True

cfg.TRAIN.TEXT_TEMPLATE = ""
# 文本,prompt,与文本模版的位置设定：
# 0：prompt+文本+ 模版
# 1：文本+模版+ prompt
cfg.TRAIN.TEXT_PROMPT_POSITION_SETTING = 0

## add
cfg.TRAIN.IMG_PROMPT_LEN = 16


# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.PROB = 0.3
# add
cfg.DATA.SAMPLER_SORT_FLAG = False
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)

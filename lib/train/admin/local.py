class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/fengxiaokun/PythonProj/tracking_proj/mmtrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/fengxiaokun/PythonProj/tracking_proj/mmtrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/fengxiaokun/PythonProj/tracking_proj/mmtrack/pretrained_networks'
        self.lasot_dir = '/home/data/wargame/fengxiaokun/LaSOT/data'

        self.tnl2k_dir = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/TNL2K_train_subset'

        self.otb_lang_dir = '/home/data1/group2/video_dataset/OTB/OTB_sentences/OTB_sentences'
        self.got10k_dir = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/got10k/train'
        self.lasot_lmdb_dir = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/got10k_lmdb'
        self.trackingnet_dir = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/trackingnet'
        self.trackingnet_lmdb_dir = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/trackingnet_lmdb'
        self.ref_coco_dir = '/home/data/wargame/fengxiaokun/refcoco'
        self.coco_lmdb_dir = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/coco_lmdb'

        self.refer_youtubevos_dir = '/home/fengxiaokun/PythonProj/tracking_proj/mmtrack/data/refer_youtubevos'


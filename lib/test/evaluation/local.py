from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.checkpoints_path = '/home/data_2/fxk_proj/python_proj/masa_track/resource'
    settings.eval_model_list = ["CTVLT_ep0055.pth.tar", "CTVLT_ep0060.pth.tar"]
    settings.save_dir = '/home/data_2/fxk_proj/python_proj/masa_track/output/tracking_res'
    settings.results_eval_dir = "/home/data_2/fxk_proj/python_proj/masa_track/output/tracking_res/baseline/tnl2k_lang_ep0060"
    settings.vl_trans_intervel = 1  #

    settings.mgit_dir = "/home/data_d/video_ds/VideoCube/VideoCube-Full"
    settings.mgit_lable_dir = "/home/data_d/video_ds/VideoCube/VideoCube-Full/VideoCube_NL/02-activity&story/"

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/got10k_lmdb'
    settings.got10k_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/itb'
    settings.lasot_extension_subset_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/lasot_lmdb'
    settings.lasot_path = '/home/data/wargame/fengxiaokun/LaSOT/data'

    settings.network_path = '/home/data_2/fxk_proj/python_proj/masa_track/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/nfs'
    settings.otb_lang_path = '/home/data1/group2/video_dataset/OTB/OTB_sentences/OTB_sentences'

    settings.otb_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/otb'
    settings.prj_dir = '/home/data_2/fxk_proj/python_proj/masa_track'

    settings.result_plot_path = '/home/data_2/fxk_proj/python_proj/masa_track/output/test/result_plots'
    settings.results_path = '/home/data_2/fxk_proj/python_proj/masa_track/output/test/tracking_results'    # Where to store tracking results

    settings.segmentation_path = '/home/data_2/fxk_proj/python_proj/masa_track/output/test/segmentation_results'
    settings.tc128_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/data/wargame/fengxiaokun/TNL2K_CVPR2021/TNL2K_test_subset'

    settings.tpl_path = ''
    settings.trackingnet_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/trackingnet'
    settings.uav_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/uav'
    settings.vot18_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/vot2018'
    settings.vot22_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/vot2022'
    settings.vot_path = '/home/data_2/fxk_proj/python_proj/masa_track/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


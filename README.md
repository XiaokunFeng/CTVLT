# CTVLT

> **Enhancing Vision-Language Tracking by Effectively
Converting Textual Cues into Visual Cues [ICASSP'25]**

[[ArXiv](https://arxiv.org/abs/2412.19648)]

## News and Updates
- 2024.12: CTVLT code is released!
- 2024.12: CTVLT is accepted by ICASSP'25!

## Overview
Vision-Language Tracking (VLT) aims to localize a target in video sequences using a visual template and language description. While textual cues enhance tracking potential, current datasets typically contain much more image data than text, limiting the ability of VLT methods to align the two modalities effectively. To address this imbalance, we propose a novel plug-and-play method named CTVLT that leverages the strong text-image alignment capabilities of foundation grounding models. CTVLT converts textual cues into interpretable visual heatmaps, which are easier for trackers to process. Specifically, we design a textual cue mapping module that transforms textual cues into target distribution heatmaps, visually representing the location described by the text. Additionally, the heatmap guidance module fuses these heatmaps with the search image to guide tracking more effectively. Extensive experiments on mainstream benchmarks demonstrate the effectiveness of our approach, achieving state-of-the-art performance and validating the utility of our method for enhanced VLT.



![motivation.jpg](asset%2Fmotivation.jpg)
Schematic diagram of motivation and method paradigm innovation. (a): Comparison of training environments between vision-language trackers and foundation grounding models. (b): The severe scarcity of textual data limits the tracker’s ability to understand text, making direct use of textual cues for guidance challenging. (c): Our core insight is to leverage the strong text-image alignment capabilities of foundation grounding models by first converting textual cues into visual cues that the tracker can easily interpret, and then using them to guide the tracker.</figcaption>


## Prepare the environment and datasets
* First, create a conda environment.
```
conda create -n ctvlt python=3.11
conda activate ctvlt
```

* Since this code is built on the [masa](https://github.com/siyuanliii/masa) and [AQATrack](https://github.com/GXNU-ZhongLab/AQATrack) repositories, 
for the installation of specific libraries, please follow the [masa](https://github.com/siyuanliii/masa)  codebase first to install the corresponding libraries; then follow the [AQATrack](https://github.com/GXNU-ZhongLab/AQATrack) codebase to install the relevant libraries.

* Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

* After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
Note: The TNL2K, LaSOT, RefCOCO, and OTB datasets are required for training; the TNL2K, LaSOT, and [MGIT](http://videocube.aitestunion.com/) datasets are required for testing.
Please ensure that the correct paths to the aforementioned datasets are provided.



## Train CTVLT
* CTVLT is derived from the [AQATrack](https://github.com/GXNU-ZhongLab/AQATrack) by incorporating the [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) model. Before training the model, it is necessary to prepare the relevant model weights.
Specifically, the pre-trained model weights for AQATrack, **AQATrack_ep150_full_384.pth.tar**, and the model weights for GroundingDINO, **gdino_masa.pth** (We use the GroundingDINO model embedded in the masa model), are required.


* The pretrained models and can be downloaded here [[Google Drive]](https://drive.google.com/drive/folders/1Stya3awPqGkCLCbZZ3ErIZeMO53g-Zsu?usp=drive_link)[[Baidu Drive]](https://pan.baidu.com/s/1yBCVZctOfGEjP6kalAcVbg?pwd=kc6m) (code: kc6m) .
Put the pretrained models in [./resource/pretrained_networks](./resource/pretrained_networks).


* Please use a command similar to the following to train the model.

```
bash train.sh
```



## Test and evaluate on benchmarks
* First, please specify the model weights to be tested and the path where the results will be saved by assigning values to the following three parameters in **lib/test/evaluation/local.py**:
```
settings.checkpoints_path: The directory of the model weights to be tested.
settings.eval_model_list: The list of model weights to be tested.
settings.save_dir: The directory where the results will be saved.
```
  For example:
```
settings.checkpoints_path = '/home/data_2/fxk_proj/python_proj/masa_track/resource/model_saved/'
settings.eval_model_list = ["CTVLT_ep0055.pth.tar", "CTVLT_ep0060.pth.tar"]
settings.save_dir = '/home/data_2/fxk_proj/python_proj/masa_track/output/tracking_res'
```
Since passing each frame through GroundingDino to obtain visual cues corresponding to text is time-consuming, 
you can control the value of `settings.vl_trans_interval` in **lib/test/evaluation/local.py** to determine the update interval for text-related visual cues (such as 10, 20, 25, etc.). Our experiments have found that increasing the update interval will slightly impair the model's performance, but it will significantly improve the speed."


* Next, run the following program to test the tracker's performance on LaSOT, TNL2K, and MGIT:
```
python tracking/test.py ctvlt baseline --dataset lasot_lang  # lasot
python tracking/test.py ctvlt baseline --dataset tnl2k_lang  # tnl2k
python tracking/test.py ctvlt baseline --dataset mgit   # mgit
```
* Finally, it's time to analyze and obtain the final evaluation results. 
For LaSOT and TNL2K, first assign the directory of the tracking results to be analyzed to `settings.results_eval_dir` in **lib/test/evaluation/local.py**,
and then run the following program to obtain the corresponding analysis results:
```
python tracking/analysis_results.py --dataset lasot_lang # lasot
python tracking/analysis_results.py --dataset tnl2k_lang # tnl2k
```
For MGIT, please upload the tracking results in the specified format to the official [website](http://videocube.aitestunion.com/).



## Model Zoo
You can download the trained models, and the raw tracking results here [[Google Drive]](https://drive.google.com/drive/folders/1Stya3awPqGkCLCbZZ3ErIZeMO53g-Zsu?usp=drive_link)[[Baidu Drive]](https://pan.baidu.com/s/1yBCVZctOfGEjP6kalAcVbg?pwd=kc6m) (code: kc6m).

## Contact
* Xiaokun Feng (email:fengxiaokun2022@ia.ac.cn)

### Acknowledgments

Our code is built on [masa](https://github.com/siyuanliii/masa), [AQATrack](https://github.com/GXNU-ZhongLab/AQATrack) and [MMTrack](https://github.com/Azong-HQU/MMTrack), thanks for their great work. If you find our work useful, consider checking out their work.



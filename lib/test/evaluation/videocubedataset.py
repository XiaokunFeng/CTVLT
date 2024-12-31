import os

import numpy as np
from lib.test.evaluation.data import Sequence,VideoCude_Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import json
import pandas as pd
from lib.test.evaluation.environment import env_settings


############
# current 00000492.png of test_015_Sord_video_Q01_done is damaged and replaced by a copy of 00000491.png
############


class VideoCubeDataset(BaseDataset):
    """
    VideoCube test set
    """

    def __init__(self, split, version='full'):
        super().__init__()

        self.split = split
        self.version = version

        f = open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'videocube.json'), 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()

        # print('sequence_list')

        self.sequence_list = self.infos[self.split]

        print('sequence_list', self.sequence_list)

        env = env_settings()
        self.base_path = env.mgit_dir

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/{}.txt'.format(self.base_path, 'attribute', 'groundtruth', sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        # text_dsp = load_str(text_dsp_path)

        nlp_path = self.base_path+'/VideoCube_NL/02-activity&story/{}.xlsx'.format(sequence_name)
        nlp_tab = pd.read_excel(nlp_path)
        nlp_rect = nlp_tab.iloc[:, [14]].values
        nlp_rect = nlp_rect[-1, 0]
        # print('nlp_rect', nlp_rect)

        frames_path = r'{}/{}/{}/{}/{}_{}'.format(self.base_path, 'data', self.split, sequence_name, 'frame', sequence_name)
        # frames_path = frames_path.replace('\\', '')
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        # target_class = class_name
        return VideoCude_Sequence(sequence_name, frames_list, 'videocube', ground_truth_rect.reshape(-1, 4), object_class=None, target_visible=None, language_query=nlp_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        path = r'{}/{}/{}_list.txt'.format(self.base_path, 'data', split)
        # path = path.replace('\\', '')
        with open(path) as f:  # list.txt
            sequence_list = f.read().splitlines()

        return sequence_list
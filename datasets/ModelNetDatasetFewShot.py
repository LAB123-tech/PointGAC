import pickle
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.builder import DATASETS
from utils.logger import *

warnings.filterwarnings('ignore')


@DATASETS.register_module()
class ModelNetFewShot(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.subset = config.subset

        self.way = config.way
        self.shot = config.shot
        self.fold = config.fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')
        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

        print('The size of %s data is %d' % (self.subset, len(self.dataset)))

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]

        if not self.use_normals:
            points = points[:, 0:3]

        pt_idx = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idx)
        current_points = points[pt_idx].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label

    def __len__(self):
        return len(self.dataset)
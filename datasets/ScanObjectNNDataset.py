import warnings

from torch.utils.data import Dataset

from datasets.dataset_utils.DatasetFunc import *
from utils.builder import DATASETS

warnings.filterwarnings('ignore')


@DATASETS.register_module()
class ScanObjectNN(Dataset):
    def __init__(self, config):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_object_dataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_object_dataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idx = np.arange(0, self.points.shape[1])
        if self.subset == 'train':
            np.random.shuffle(pt_idx)
        current_points = self.points[idx, pt_idx].copy()
        current_points = torch.from_numpy(current_points).float()
        label = torch.tensor(self.labels[idx]).long()
        return current_points, label

    def __len__(self):
        return self.points.shape[0]


@DATASETS.register_module()
class ScanObjectNN_hardest(Dataset):
    def __init__(self, config):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_object_dataset_augmented_rot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_object_dataset_augmented_rot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idx = np.arange(0, self.points.shape[1])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idx)
        current_points = self.points[idx, pt_idx].copy()
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        return current_points, label

    def __len__(self):
        return self.points.shape[0]

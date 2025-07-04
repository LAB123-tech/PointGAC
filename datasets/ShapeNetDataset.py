import torch.utils.data as data

from datasets.dataset_utils.DatasetFunc import *
from utils.builder import DATASETS


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_path = config.DATA_PATH
        self.subset = config.subset
        self.data_list_file = os.path.join(self.data_path, self.subset + "_my.txt")

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        self.file_list = []
        for line in lines:
            line = line.strip()
            self.file_list.append({'file_path': line})

        print(f'[Dataset] {config.subset} {len(self.file_list)} instances were loaded')

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc_data = IO.get(os.path.join(self.data_path, sample['file_path'])).astype(np.float32)
        return pc_data

    def __len__(self):
        return len(self.file_list)

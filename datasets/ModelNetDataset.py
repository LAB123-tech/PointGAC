import warnings
from pathlib import Path

from torch.utils.data import Dataset

from datasets.dataset_utils.DatasetFunc import *
from utils.builder import DATASETS

warnings.filterwarnings('ignore')


@DATASETS.register_module()
class ModelNet(Dataset):
    def __init__(self, config):
        self.root = Path(config.PC_PATH)
        self.subset = config.subset
        self.subset_data_path = [path / self.subset for path in self.root.glob("*") if path.is_dir()]
        self.class_name = sorted([path.parent.name for path in self.subset_data_path])
        self.files = load_filenames(self.subset_data_path)
        self.labels = load_labels(self.files, self.class_name)

        print(f'[Dataset] {config.subset} {len(self.files)} instances were loaded')

    def __getitem__(self, index):
        path = self.files[index]
        label = self.labels[index]
        data = IO.get(path)
        data = torch.from_numpy(data).float()
        label = torch.tensor(label).long()
        return data, label

    def __len__(self):
        return len(self.files)

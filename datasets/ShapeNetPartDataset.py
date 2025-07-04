import json
from torch.utils.data import Dataset
from utils.builder import DATASETS
from datasets.dataset_utils.DatasetFunc import *
from utils.logger import log_string


@DATASETS.register_module()
class ShapeNetPart(Dataset):
    def __init__(self, config):
        self.cat = {}
        self.root = config.DATA_PATH
        self.split = config.subset
        self.normal_channel = config.USE_NORMALS
        self.catfile = os.path.join(self.root, 'class2file.txt')

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if self.split == 'train_val':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif self.split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif self.split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif self.split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % self.split)
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
                            'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
                            'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29],
                            'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
                            'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}

        print(f'[Dataset] {self.split} {len(self.datapath)} instances were loaded')

    def __getitem__(self, index):
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)
        data = IO.get(fn[1]).astype(np.float32)
        if self.normal_channel:
            point_set = data[:, 0:6]
        else:
            point_set = data[:, 0:3]
        seg = data[:, -1].astype(np.int32)
        choice = np.random.choice(len(seg), 2400, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice][:, None]
        data = np.concatenate([point_set, seg], axis=-1)
        return data, cls

    def __len__(self):
        return len(self.datapath)

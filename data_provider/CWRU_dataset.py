import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class Dataset_CWRU(Dataset):
    def __init__(self, args, flag):
        self.flag = flag.upper()
        self.df = args.train_data if self.flag == 'TRAIN' else args.test_data
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.features = args.features
        self.use_norm = args.use_norm

        self.X = self.df[:, 1:].astype(np.float32).reshape(-1, self.seq_len, self.enc_in)
        self.y = self.df[:, 0].astype(np.int64)

        # ✅ 标准化处理（Z-score）：均值为0，方差为1
        self.scaler = None
        if self.use_norm:
            self.scaler = np.zeros((self.enc_in, 2))  # 存均值和标准差
            for i in range(self.enc_in):
                mean = np.mean(self.X[:, :, i])
                std = np.std(self.X[:, :, i]) + 1e-8  # 防止除0
                self.scaler[i, 0] = mean
                self.scaler[i, 1] = std
                self.X[:, :, i] = (self.X[:, :, i] - mean) / std

        print(f"{self.flag} dataset: X shape {self.X.shape}, y shape {self.y.shape}, "
              f"classes {len(np.unique(self.y))}, normalization: {self.use_norm}")

        self.max_seq_len = self.seq_len
        self.class_names = list(map(str, np.unique(self.y)))
        self.feature_df = pd.DataFrame(self.X.reshape(len(self.X), -1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "seq_x": torch.from_numpy(self.X[idx]).float(),
            "labels": torch.tensor(self.y[idx]).long()
        }

    def inverse_transform(self, data):
        """标准化的逆变换"""
        if self.scaler is None or not self.use_norm:
            return data
        # 广播恢复原始值
        return data * self.scaler[:, 1][:, None, None] + self.scaler[:, 0][:, None, None]


def build_dataset(data_set, args):
    shuffle = args.task_name == 'classification' and data_set.flag == 'TRAIN'
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True
    )
    return data_loader

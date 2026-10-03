'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.data_utils import load_splits, prepare_users, user_targets, check_features


class PopularityDataset(Dataset):
    def __init__(self, args, phase, preprocessing=None):
        root = Path(args.data_root)
        target = np.load(root / 'target.npy')[:, 0]
        splits, hashes = load_splits(args.splits_root, args.dataset, args.split_idx, len(target))
        self.indices = splits[phase]
        self.transform = args.target_transform
        if self.transform == 'log':
            if np.any(target <= 0):
                raise ValueError('Log popularity targets must be positive')
            transformed = np.log(target)
        else:
            transformed = np.asarray(target, dtype=np.float64)
        if preprocessing is None:
            train_target = transformed[splits['train']]
            target_state = dict(mean=float(train_target.mean()), std=float(train_target.std()))
            user_state = None
        else:
            if preprocessing['split_hashes'] != hashes:
                raise ValueError('Checkpoint and data splits differ')
            target_state, user_state = preprocessing['target'], preprocessing['user']
        self.target_mean, self.target_std = target_state['mean'], target_state['std']
        if self.target_std <= 0 or not np.isfinite(self.target_std):
            raise ValueError('Invalid target variance')
        self.target = ((transformed[self.indices] - self.target_mean) / self.target_std).astype(np.float32)
        users = np.load(root / 'user.npy')
        if users.shape != (len(target), args.user_dim):
            raise ValueError('User feature shape does not match the dataset configuration')
        self.users, self.user_state = prepare_users(users, splits['train'], user_state)
        self.features = {name: np.load(root / f'{filename}.npy', mmap_mode='r')
                         for name, filename in args.mod2feat_dict.items()}
        check_features(self.features, args.mod2dim_dict, len(target))
        ids = pd.read_csv(root / 'vuid_list.txt', sep='\t', header=None, dtype=str)
        if ids.shape != (len(target), 2):
            raise ValueError('vuid_list.txt must align with feature rows')
        self.user_pop = user_targets(ids.iloc[:, 1].to_numpy(), target, splits['train'],
                                     self.target_mean, self.target_std, self.transform)
        self.preprocessing = dict(target=target_state, user=self.user_state, split_hashes=hashes)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return dict(u_feat=torch.tensor(self.users[index], dtype=torch.float32),
                    v_feat={name: torch.tensor(value[index], dtype=torch.float32)
                            for name, value in self.features.items()},
                    tgt_pop=torch.tensor(self.target[position]),
                    tgt_mean_pop=torch.tensor(self.user_pop[index]),
                    index=torch.tensor(index))

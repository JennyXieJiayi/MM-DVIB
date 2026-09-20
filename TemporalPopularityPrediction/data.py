'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.data_utils import load_splits, prepare_users, user_targets, check_features


CATEGORY_IDS = {'Travel  Events': 0, 'People  Blogs': 1, 'Gaming': 2,
                'News  Politics': 3, 'Entertainment': 4, 'Music': 5,
                'Education': 6, 'Sports': 7, 'Howto  Style': 8,
                'Film  Animation': 9, 'Nonprofits  Activism': 10,
                'Travel': 11, 'Comedy': 12, 'Science  Technology': 13,
                'Autos  Vehicles': 14, 'Pets  Animals': 15}


class TemporalPopularityDataset(Dataset):
    def __init__(self, args, phase, preprocessing=None):
        root = Path(args.data_root)
        self.dataset = args.dataset
        if args.dataset == 'Xigua':
            target_array = np.load(root / f'target/len_{args.pop_len}/target.npy', mmap_mode='r')
            target = np.asarray(target_array[:, :, 0], dtype=np.float64)
        else:
            target = np.load(root / 'target_first9_daily_view_count_increment_log2p1.npy', mmap_mode='r')
        if target.ndim != 2 or target.shape[1] != args.pop_len:
            raise ValueError('Target sequence length does not match pop_len')
        splits, hashes = load_splits(args.splits_root, args.dataset, args.split_idx, len(target))
        self.indices = splits[phase]
        if preprocessing is None:
            train_target = np.asarray(target[splits['train']], dtype=np.float64)
            target_state = dict(mean=float(train_target.mean()), std=float(train_target.std()))
            user_state = None
        else:
            if preprocessing['split_hashes'] != hashes:
                raise ValueError('Checkpoint and data splits differ')
            target_state, user_state = preprocessing['target'], preprocessing['user']
        self.target_mean, self.target_std = target_state['mean'], target_state['std']
        if self.target_std <= 0 or not np.isfinite(self.target_std):
            raise ValueError('Invalid target variance')
        self.target = ((target[self.indices] - self.target_mean) / self.target_std).astype(np.float32)
        users = np.load(root / 'user.npy')
        if users.shape != (len(target), args.user_dim):
            raise ValueError('User feature shape does not match the dataset configuration')
        self.users, self.user_state = prepare_users(users, splits['train'], user_state)
        self.features = {name: np.load(root / f'{filename}.npy', mmap_mode='r')
                         for name, filename in args.mod2feat_dict.items()}
        check_features(self.features, args.mod2dim_dict, len(target))
        if args.dataset == 'Xigua':
            ids = pd.read_csv(root / 'vuid_list.txt', header=None, dtype=str)
            if ids.shape != (len(target), 2):
                raise ValueError('vuid_list.txt must align with feature rows')
            user_ids = ids.iloc[:, 1].to_numpy()
            self.time = target_array[self.indices, :, -1]
        else:
            user_ids = users[:, 0].astype(np.int64)
            days = np.arange(1, 31, dtype=np.float32)
            self.time = ((days - days.mean()) / days.std())[:9]
            category_map = json.loads((root / 'category_mapping.json').read_text())
            lookup = np.empty(len(category_map), dtype=np.int64)
            for name, cached_id in category_map.items():
                lookup[cached_id] = CATEGORY_IDS[name]
            self.category = lookup[np.load(root / 'category.npy')[:, 0]]
            self.language = np.load(root / 'language.npy')[:, 0]
            self.metadata = np.load(root / 'content_metadata.npy', mmap_mode='r')
            cfg = args.structured_config
            if (self.category.shape != (len(target),) or self.language.shape != (len(target),)
                    or self.metadata.shape != (len(target), cfg['metadata_dim'])
                    or self.category.min() < 0 or self.category.max() >= cfg['category_count']
                    or self.language.min() < 0 or self.language.max() >= cfg['language_count']):
                raise ValueError('Invalid SMTPD structured feature shapes or IDs')
        self.user_pop = user_targets(user_ids, target, splits['train'])
        self.preprocessing = dict(target=target_state, user=self.user_state, split_hashes=hashes)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        content = {name: torch.tensor(value[index], dtype=torch.float32)
                   for name, value in self.features.items()}
        if self.dataset == 'SMTPD':
            content['structured'] = dict(category=torch.tensor(self.category[index]),
                                         language=torch.tensor(self.language[index]),
                                         metadata=torch.tensor(self.metadata[index], dtype=torch.float32))
        time = self.time[position] if self.dataset == 'Xigua' else self.time
        return dict(u_feat=torch.tensor(self.users[index], dtype=torch.float32), v_feat=content,
                    time=torch.tensor(time, dtype=torch.float32),
                    tgt_pop=torch.tensor(self.target[position]),
                    tgt_mean_pop=torch.tensor(self.user_pop[index]), index=torch.tensor(index))

'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import hashlib
from pathlib import Path

import numpy as np


def load_splits(root, dataset, split, count):
    paths = [Path(root) / dataset / str(split) / f'{phase}.txt'
             for phase in ('train', 'val', 'test')]
    indices = [np.loadtxt(p, dtype=np.int64, ndmin=1) for p in paths]
    joined = np.concatenate(indices)
    if (any(len(i) == 0 for i in indices) or joined.min() < 0 or joined.max() >= count
            or len(np.unique(joined)) != len(joined)):
        raise ValueError('Splits must contain disjoint, valid feature-row indices')
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    return dict(zip(('train', 'val', 'test'), indices)), hashes


def user_statistics(values, indices):
    mean = np.zeros(values.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    total = 0
    for start in range(0, len(indices), 32768):
        block = np.asarray(values[indices[start:start + 32768]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Non-finite user attributes')
        block_mean = block.mean(axis=0)
        block_m2 = np.square(block - block_mean).sum(axis=0)
        n = len(block)
        delta = block_mean - mean
        m2 += block_m2 + delta ** 2 * total * n / (total + n)
        mean += delta * n / (total + n)
        total += n
    std = np.sqrt(m2 / total)
    std[std < 1e-8] = 1.0  # Constant columns become zero after centering.
    return mean, std


def prepare_users(user, train_indices, state=None):
    user = np.asarray(user, dtype=np.float32)
    ids = user[:, 0].astype(np.int64)
    if np.any(ids < 0) or not np.array_equal(ids, user[:, 0]):
        raise ValueError('User IDs must be non-negative integers')
    if state is None:
        mean, std = user_statistics(user[:, 1:], train_indices)
        state = dict(mean=mean.tolist(), std=std.tolist(),
                     known_ids=np.unique(ids[train_indices]).tolist(),
                     padding_idx=int(ids.max()) + 1)
    mean, std = np.asarray(state['mean'], np.float32), np.asarray(state['std'], np.float32)
    mapped = ids.copy()
    mapped[~np.isin(ids, state['known_ids'])] = state['padding_idx']
    attributes = (user[:, 1:] - mean) / std
    return np.column_stack((mapped.astype(np.float32), attributes)), state


def user_targets(ids, target, train_indices, target_mean=None, target_std=None, transform='identity'):
    ids = np.asarray(ids)
    unique, inverse = np.unique(ids[train_indices], return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(unique),) + target.shape[1:], dtype=np.float64)
    np.add.at(sums, inverse, target[train_indices])
    means = sums / counts.reshape((-1,) + (1,) * (target.ndim - 1))
    if target_mean is None:
        target_mean, target_std = float(means.mean()), float(means.std())
    if transform == 'log':
        means = np.log(means)
    if target_std <= 0 or not np.isfinite(target_std):
        raise ValueError('Invalid user-target variance')
    means = (means - target_mean) / target_std
    positions = np.searchsorted(unique, ids)
    known = positions < len(unique)
    known[known] &= unique[positions[known]] == ids[known]
    result = np.zeros((len(ids),) + target.shape[1:], dtype=np.float32)
    result[known] = means[positions[known]]
    return result


def check_features(features, dimensions, count):
    for name, values in features.items():
        if values.shape != (count, dimensions[name]):
            raise ValueError(f'{name} feature shape is {values.shape}; expected {(count, dimensions[name])}')


def model_config(args, data):
    config = dict(num_u=data.user_state['padding_idx'] + 1,
                  u_in_size=args.user_dim, u_emb_size=args.user_emb_dim,
                  dec_type=args.dec_type, hid_size=args.hid_size,
                  mod_in_sizes=args.mod2dim_dict, modalities=args.modalities,
                  drop_p=args.dropout, user_padding_idx=data.user_state['padding_idx'])
    if args.dataset in ('Xigua', 'SMTPD'):
        config['t_emb_size'] = args.time_emb_dim
        config['structured_config'] = args.structured_config if args.dataset == 'SMTPD' else None
    return config

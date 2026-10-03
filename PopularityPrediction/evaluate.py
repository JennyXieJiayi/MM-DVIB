'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from utils.utils import model_inputs


def calculate_metrics(predictions, targets):
    predictions, targets = np.asarray(predictions, np.float64), np.asarray(targets, np.float64)
    if predictions.shape != targets.shape or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError('Predictions and targets must have matching shapes and finite values')
    variance = targets.var()
    if variance <= 0:
        raise ValueError('nMSE is undefined for constant targets')
    metrics = dict(nmse=float(np.mean((predictions - targets) ** 2) / variance), sample_count=len(targets))
    if targets.ndim == 1:
        metrics['srcc'] = float(spearmanr(predictions, targets).statistic)
    else:
        p = predictions - predictions.mean(axis=1, keepdims=True)
        y = targets - targets.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1)
        valid = denominator > 0
        if not valid.any():
            raise ValueError('PLCC is undefined for all target/prediction sequences')
        correlations = (p[valid] * y[valid]).sum(axis=1) / denominator[valid]
        metrics['plcc'] = float(correlations.mean())
        metrics['valid_plcc_samples'] = int(valid.sum())
    return metrics


@torch.no_grad()
def evaluate(model, data, batch_size, device, num_workers=0):
    model.eval()
    predictions, targets, indices = [], [], []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0)
    for batch in loader:
        predictions.append(model.predict(*model_inputs(batch, device)).cpu().numpy())
        targets.append(batch['tgt_pop'].numpy())
        indices.append(batch['index'].numpy())
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    arrays = dict(sample_indices=np.concatenate(indices), predictions_norm=predictions, targets_norm=targets)
    if targets.ndim == 1:
        predictions = predictions.astype(np.float64) * data.target_std + data.target_mean
        targets = targets.astype(np.float64) * data.target_std + data.target_mean
        if data.transform == 'log':
            predictions, targets = np.exp(predictions), np.exp(targets)
        arrays.update(predictions_raw=predictions, targets_raw=targets)
    return calculate_metrics(predictions, targets), arrays

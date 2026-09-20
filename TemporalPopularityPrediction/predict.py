'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import TemporalPopularityDataset
from evaluate import evaluate
from model import DMMVED
from utils.parser import parse_args, DATASETS
from utils.utils import seed_everything


def predict(args):
    path = Path(args.trained_model_path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    training = argparse.Namespace(**checkpoint['train_args'])
    if training.dataset not in DATASETS:
        raise ValueError('Checkpoint belongs to the other prediction task')
    if args.dataset is not None and args.dataset != training.dataset:
        raise ValueError('Requested dataset differs from the checkpoint')
    seed_everything(training.seed)
    training.data_root, training.splits_root = args.data_root, args.splits_root
    data = TemporalPopularityDataset(training, 'test', checkpoint['preprocessing'])
    device = torch.device(args.device)
    model = DMMVED(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics, arrays = evaluate(model, data, args.test_batch_size, device, args.num_workers)
    output = Path(args.output_dir) if args.output_dir else path.parent / 'test'
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'test_predictions.npz', **arrays)
    result = dict(dataset=training.dataset, split=training.split_idx, epoch=checkpoint['epoch'], metrics=metrics)
    (output / 'test_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    predict(parse_args(prediction=True))

'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['Xigua', 'SMTPD']


def parse_args(prediction=False):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--dataset', choices=DATASETS, default=None if prediction else DATASETS[0])
    dataset = selector.parse_known_args()[0].dataset
    config = {} if prediction else json.loads((ROOT / 'configs' / f'{dataset}.json').read_text())
    parser = argparse.ArgumentParser(parents=[selector])
    parser.add_argument('--data_root', default='./data', help='Directory containing prepared feature arrays')
    parser.add_argument('--splits_root', default=str(ROOT / 'splits'))
    parser.add_argument('--device', default='cpu', help='cpu or cuda:N')
    parser.add_argument('--num_workers', type=int, default=0)
    if prediction:
        parser.add_argument('--trained_model_path', '--checkpoint', required=True)
        parser.add_argument('--output_dir', default=None)
        parser.add_argument('--test_batch_size', type=int, default=512)
    else:
        parser.add_argument('--split_idx', type=int, choices=range(1, 6), default=1)
        parser.add_argument('--save_dir', default=None)
        parser.add_argument('--seed', type=int, default=888)
        parser.add_argument('--epoch_num', type=int, default=100)
        parser.add_argument('--train_batch_size', type=int)
        parser.add_argument('--test_batch_size', type=int)
        parser.add_argument('--lambda_u', type=float)
        parser.add_argument('--lambda_v', type=float)
        parser.add_argument('--factor_uv', type=float)
        parser.add_argument('--content_coefficient_min', type=float)
        parser.add_argument('--user_emb_dim', type=int)
        parser.add_argument('--hid_size', type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--weight_decay', type=float)
        parser.add_argument('--peak_lr', type=float, default=1 / (8 * 4000) ** 0.5)
        parser.add_argument('--warmup_ratio', type=float, default=0.1)
        parser.add_argument('--min_lr_ratio', type=float, default=0.01)
        if dataset in ('Xigua', 'SMTPD'):
            parser.add_argument('--pop_len', type=int, choices=[6, 9, 12, 18, 36] if dataset == 'Xigua' else [9])
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not prediction:
        if (args.epoch_num < 1 or min(args.train_batch_size, args.test_batch_size) < 1
                or not 0 <= args.content_coefficient_min <= 1
                or not 0 <= args.dropout < 1 or not 0 < args.warmup_ratio <= 1
                or not 0 <= args.min_lr_ratio <= 1 or args.peak_lr <= 0):
            parser.error('Invalid training parameter range')
    return args

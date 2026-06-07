'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import argparse
import time
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--start_epoch_idx', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--opt_factor', type=float, default=10)
    parser.add_argument('--opt_model_size', type=int, default=512)
    parser.add_argument('--opt_warmup', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=4,
                        help='Iteration interval of printing loss.')
    parser.add_argument('--save_every', type=int, default=999,
                        help='Iteration interval of saving model.')
    parser.add_argument('--evaluate_every', type=int, default=4,
                        help='Epoch interval of evaluation.')
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument('--pre_model_path', type=str, default='')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--cuda_idx', type=int, default=0)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument("--dec_type", type=str, default="ffn", choices=["ffn"])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--lambda_u', type=float, default=0.4, choices=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    parser.add_argument('--lambda_v', type=float, default=0.2, choices=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    parser.add_argument('--factor_uv', type=float, default=0.2, choices=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    parser.add_argument('--user_emb_dim', type=int, default=4)
    parser.add_argument('--hid_size', type=int, default=16)
    parser.add_argument('--modalities', type=list, default=['visual', 'aural', 'textual'])

    args = parser.parse_args()
    args.mod2feat_dict = {
        'visual': 'visual',
        'aural': 'aural',
        'textual': 'textual'
    }
    args.mod2abbr_dict = {
        'visual': 'V',
        'aural': 'A',
        'textual': 'T',
    }
    args.mod2dim_dict = {
        'visual': 128,
        'aural': 128,
        'textual': 101,
    }

    args.mod_features = [feat for mod, feat in args.mod2feat_dict.items() if mod in args.modalities]
    args.mod_abbreviations = ''.join(abbr for mod, abbr in args.mod2abbr_dict.items())
    args.save_dir = os.path.join('models', '{}'.format(time.strftime("%Y%m%d_%H%M%S")), args.mod_abbreviations)
    args.num_user = 11057
    args.user_dim = 7

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args.modalities)

'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import TemporalPopularityDataset
from evaluate import evaluate
from model import DMMVED
from utils.checkpoints import BestCheckpoint
from utils.data_utils import model_config
from utils.optimizer import WarmupCosine
from utils.parser import parse_args
from utils.utils import seed_everything, model_inputs, project_content_coefficient


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path('models') / args.dataset
    if args.dataset == 'Xigua':
        output /= f'len{args.pop_len}'
    output = Path(args.save_dir) if args.save_dir else output / f'split{args.split_idx}'
    output.mkdir(parents=True, exist_ok=False)
    train_data = TemporalPopularityDataset(args, 'train')
    val_data = TemporalPopularityDataset(args, 'val', train_data.preprocessing)
    loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                        persistent_workers=args.num_workers > 0)
    config = model_config(args, train_data)
    model = DMMVED(**config).to(device)
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    with torch.no_grad():
        model.u_encoder.user_emb.weight[config['user_padding_idx']].zero_()
    project_content_coefficient(model, args.content_coefficient_min)
    optimizer_class = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam}[args.optimizer_name]
    base_optimizer = optimizer_class(model.parameters(), lr=0, betas=(0.9, 0.98),
                                     eps=1e-9, weight_decay=args.weight_decay)
    optimizer = WarmupCosine(base_optimizer, len(loader) * args.epoch_num, args.peak_lr,
                            args.warmup_ratio, args.min_lr_ratio)
    joint = args.dataset in ('Vine', 'ICIP', 'Instagram')
    selection = BestCheckpoint(joint=joint)
    correlation = 'srcc' if joint else 'plcc'
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    fields = ['epoch', 'train_loss', 'val_nmse', f'val_{correlation}', 'learning_rate', 'content_coefficient']
    with (output / 'train_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epoch_num + 1):
            model.train()
            loss_sum = 0.0
            for batch in loader:
                base_optimizer.zero_grad()
                mean_pred, prediction = model(*model_inputs(batch, device))
                loss, *_ = model.loss(mean_pred, prediction, batch['tgt_mean_pop'].to(device),
                                      batch['tgt_pop'].to(device), args.lambda_u, args.lambda_v, args.factor_uv)
                if not torch.isfinite(loss):
                    raise ValueError(f'Non-finite training loss at epoch {epoch}')
                loss.backward()
                lr = optimizer.step()
                project_content_coefficient(model, args.content_coefficient_min)
                loss_sum += float(loss.detach())
            metrics, _ = evaluate(model, val_data, args.test_batch_size, device, args.num_workers)
            selection.update(model, epoch, metrics)
            writer.writerow(dict(epoch=epoch, train_loss=loss_sum / len(loader), val_nmse=metrics['nmse'],
                                 **{f'val_{correlation}': metrics[correlation]}, learning_rate=lr,
                                 content_coefficient=float(model.decoder.uv_param.detach())))
            handle.flush()
            print(f"Epoch {epoch}/{args.epoch_num}: val nMSE={metrics['nmse']:.6f}, {correlation}={metrics[correlation]:.6f}", flush=True)
    best = selection.best()
    path = output / ('best_joint.pt' if joint else 'best_PLCC.pt')
    torch.save(dict(model_state_dict=best['state'], model_config=config, train_args=vars(args),
                    preprocessing=train_data.preprocessing, epoch=best['epoch'], val_metrics=best['metrics']), path)
    print(f"Selected epoch {best['epoch']}: {path}")
    return path


if __name__ == '__main__':
    train(parse_args())

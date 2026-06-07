'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import os
from collections import OrderedDict
import torch


def save_model(model, model_dir, current_epoch, last_best_epoch=None, tag=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if last_best_epoch is not None and current_epoch != last_best_epoch:
        if tag:
            model_state_file = os.path.join(model_dir, 'model_best_{}.pth'.format(tag))
        else:
            model_state_file = os.path.join(model_dir, 'model_best.pth')
    else:
        model_state_file = os.path.join(model_dir, 'model_{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'], False)
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]
            # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def degrade_saved_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    save_path = os.path.join(os.path.dirname(model_path), 'degrade_version')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(checkpoint, os.path.join(save_path, os.path.basename(model_path)), _use_new_zipfile_serialization=False)

'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_inputs(batch, device):
    user = batch['u_feat'].to(device)
    content = {name: ({key: item.to(device) for key, item in value.items()}
                      if isinstance(value, dict) else value.to(device))
               for name, value in batch['v_feat'].items()}
    inputs = (user, content)
    if 'time' in batch:
        inputs += (batch['time'].to(device),)
    return inputs


def project_content_coefficient(model, minimum):
    with torch.no_grad():
        model.decoder.uv_param.clamp_(min=minimum)

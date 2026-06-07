'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import torch
from torch.utils.data import DataLoader


def evaluate(trained_model, data, batch_size, use_cuda=True, device='cpu'):
    trained_model.eval()
    total_sample_num = len(data)
    pop_log_mean = data.target_log_mean
    pop_log_std = data.target_log_std
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    tgt_pop = np.empty((total_sample_num), dtype=np.float32)
    out_pop = np.empty((total_sample_num), dtype=np.float32)
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            u_feat = batch_data['u_feat'].squeeze().to(device)
            v_feat = batch_data['v_feat']
            for mod_key in v_feat.keys():
                v_feat[mod_key] = v_feat[mod_key].to(device)
            batch_tgt_pop = batch_data['tgt_pop'].squeeze()
            batch_out_pop = trained_model.predict(u_feat, v_feat)
            if use_cuda:
                batch_out_pop = batch_out_pop.cpu()
            true_batch_size = batch_data['tgt_pop'].shape[0]
            tgt_pop[idx*batch_size:idx*batch_size+true_batch_size] = batch_tgt_pop
            out_pop[idx*batch_size:idx*batch_size+true_batch_size] = batch_out_pop

    out_pop = np.exp(out_pop * pop_log_std + pop_log_mean)
    tgt_pop = np.exp(tgt_pop * pop_log_std + pop_log_mean)
    nmse = cal_nmse(out_pop, tgt_pop)
    plcc, _ = pearsonr(out_pop, tgt_pop)
    srcc, p_val = spearmanr(out_pop, tgt_pop)
    return nmse, plcc, srcc, p_val


def cal_nmse(preds, truth):
    return np.mean(np.square(preds - truth)) / (truth.std() ** 2)


if __name__ == "__main__":
    # for test only
    from data import XiguaDataset
    mod2feat_dict = {
        'visual': 'resnet50',
        'aural': 'audiovgg',
        'textual': 'fudannlp'
    }
    data4test = XiguaDataset('test', 9, './data', 10, ['visual', 'textual'], mod2feat_dict)
    result = evaluate(None, data4test, 256)
    print(['%.5f'%i for i in result])

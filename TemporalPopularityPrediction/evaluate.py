'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader


def evaluate(trained_model, data, batch_size, use_cuda=True, device='cpu'):
    trained_model.eval()
    total_sample_num = len(data)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    tgt_pop = np.empty([total_sample_num, data.pop_len], dtype=np.float32)
    out_pop = np.empty([total_sample_num, data.pop_len], dtype=np.float32)
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            u_feat = batch_data['u_feat'].squeeze().to(device)
            v_feat = batch_data['v_feat']
            for mod_key in v_feat.keys():
                v_feat[mod_key] = v_feat[mod_key].to(device)
            time_pop = batch_data['time'].squeeze().to(device)
            batch_tgt_pop = batch_data['tgt_pop'].squeeze()
            batch_out_pop = trained_model.predict(u_feat, v_feat, time_pop)
            if use_cuda:
                batch_out_pop = batch_out_pop.cpu()
            true_batch_size = batch_data['tgt_pop'].shape[0]
            tgt_pop[idx*batch_size:idx*batch_size+true_batch_size] = batch_tgt_pop
            out_pop[idx*batch_size:idx*batch_size+true_batch_size] = batch_out_pop

    nmse = cal_nmse(out_pop, tgt_pop)
    plcc = cal_pearson_corr(out_pop, tgt_pop)
    srcc, p_val = cal_spearman_corr(out_pop, tgt_pop)
    return nmse, plcc, srcc, p_val


def cal_nmse(preds, truth):
    return np.mean(np.square(preds - truth)) / (truth.std()**2)


def cal_spearman_corr(preds, truth):
    corr = 0
    p_val = 0
    num_samples = len(preds)
    cnt_samples = num_samples
    for i in range(num_samples):
        corr_this, p_value_this = spearmanr(pd.Series(preds[i]), pd.Series(truth[i]))
        if np.isnan(corr_this):
            cnt_samples = cnt_samples - 1
            continue
        corr += corr_this
        p_val += p_value_this
    return corr / cnt_samples, p_val / cnt_samples


def cal_pearson_corr(preds, truth):
    corr = 0
    num_samples = len(preds)
    cnt_samples = num_samples
    for i in range(num_samples):
        corr_this = pd.Series(preds[i]).corr(pd.Series(truth[i]), method='pearson')
        if np.isnan(corr_this):
            cnt_samples = cnt_samples - 1
            continue
        corr += corr_this
    return corr / cnt_samples


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

'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class XiguaDataset(Dataset):
	def __init__(self, phase, pop_len, data_root, split_idx, modalities, mod2feat_dict, time_emb=False, logging=None, num_test_neg=None):
		assert phase in ['train', 'val', 'test']

		self.pop_len = pop_len
		train_idx_path = os.path.join(data_root, 'split/{}'.format(split_idx), 'train.txt')
		train_idx = pd.read_table(train_idx_path, header=None).values.squeeze()

		phase_idx_path = os.path.join(data_root, 'split/{}'.format(split_idx), '{}.txt'.format(phase))
		phase_idx = pd.read_table(phase_idx_path, header=None).values.squeeze()

		vuid_file = os.path.join(data_root, 'vuid_list.txt')
		vuids_all = pd.read_table(vuid_file, header=None, dtype=str, sep=',')
		vuids_all.columns = ['vid', 'uid']

		target_file = os.path.join(data_root, 'target', 'len_{}'.format(pop_len), 'target.npy')
		target_unnorm_all = np.load(target_file)[:, :, 0] # unnorm pop for mean pop
		self.target = np.load(target_file)[phase_idx, :, -2] # norm pop

		self.time_emb = time_emb
		if time_emb:
			self.time = np.load(target_file)[phase_idx, :, 1]  # unnorm time
		else:
			self.time = np.load(target_file)[phase_idx, :, -1] # norm time

		u_feat_file = os.path.join(data_root, 'user.npy')
		self.u_feat = np.load(u_feat_file)[phase_idx]

		self.v_feat = {}
		for modality in modalities:
			mod_file = os.path.join(data_root, "{}.npy".format(mod2feat_dict[modality]))
			self.v_feat[modality] = np.load(mod_file)[phase_idx]

		self.mean_pop = self.get_mean_pop(vuids_all, target_unnorm_all, train_idx, phase_idx) # norm pop

	def get_mean_pop(self, vuids, target, train_idxes, phase_idxes):
		### use train set to calculate mean pop
		### new users in test / val use zero mean pop

		# 1. calculate the mean pop of users in train set (for train&val) or train/val set (for test)
		uids_known = vuids['uid'][train_idxes].tolist()
		tgt_known = target[train_idxes]
		unique_uids = list(set(uids_known))
		uids_vcount = np.zeros(len(unique_uids))
		uids_sum_pop = np.zeros((len(unique_uids), target.shape[1]))
		for idx, uid in enumerate(uids_known):
			uids_vcount[unique_uids.index(uid)] += 1
			uids_sum_pop[unique_uids.index(uid)] += tgt_known[idx,:]
		uids_mean_pop = uids_sum_pop / uids_vcount.reshape(-1, 1)
		uids_mean_pop = (uids_mean_pop - uids_mean_pop.mean()) / uids_mean_pop.std() # normalization
		uids_mean_pop_missing = np.zeros_like(uids_mean_pop[0])

		# 2. map to the phase set
		vids_mean_pop = np.zeros((len(phase_idxes), target.shape[1]))
		vids_unknown = vuids['vid'][phase_idxes].tolist()
		uids_unknown = vuids['uid'][phase_idxes].tolist()
		for idx, (vid, uid) in enumerate(zip(vids_unknown, uids_unknown)):
			if uid in unique_uids:
				vids_mean_pop[idx] = uids_mean_pop[unique_uids.index(uid)]
			else:
				vids_mean_pop[idx] = uids_mean_pop_missing
		return vids_mean_pop

	def __len__(self):
		return len(self.target)

	def __getitem__(self, index):
		if self.time_emb:
			sample_time = torch.tensor([self.time[index]], dtype=torch.long)
		else:
			sample_time = torch.tensor([self.time[index]], dtype=torch.float32)

		samples = {
			'u_feat': torch.tensor([self.u_feat[index]], dtype=torch.float32),
			'v_feat': {key: torch.tensor(mod[index], dtype=torch.float32) for key, mod in self.v_feat.items()},
			'time': sample_time,
			'tgt_pop': torch.tensor([self.target[index]], dtype=torch.float32),
			'tgt_mean_pop': torch.tensor([self.mean_pop[index]], dtype=torch.float32)
		}
		return samples


if __name__ == "__main__":
	# for test only
	modalities = ['visual', 'aural', 'textual']
	mod2feat_dict = {
		'visual': 'resnet50',
		'aural': 'audiovgg',
		'textual': 'fudannlp'
	}
	data4test = XiguaDataset('train', 9, './data/', 10, modalities, mod2feat_dict)
	print(len(data4test)) # total sample num
	dataloader4test = DataLoader(data4test, batch_size=128, shuffle=True)
	print(len(dataloader4test)) # batch num
	for i_batch, batch_data in enumerate(dataloader4test):
		print(batch_data['u_feat'].shape)
		print(batch_data['v_feat'].keys())
		print(batch_data['v_feat']['visual'].shape)
		print(batch_data['v_feat']['aural'].shape)
		print(batch_data['v_feat']['textual'].shape)
		print(batch_data['time'].shape)
		print(batch_data['tgt_pop'].shape)
		print(batch_data['tgt_mean_pop'].shape)
		break
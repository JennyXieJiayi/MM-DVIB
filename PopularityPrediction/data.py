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
	def __init__(self, phase, data_root, split_idx, modalities, mod2feat_dict, logging=None, num_test_neg=None):
		assert phase in ['train', 'val', 'test']

		train_idx_path = os.path.join(data_root, 'split/{}'.format(split_idx), 'train.txt')
		train_idx = pd.read_table(train_idx_path, header=None).values.squeeze()

		phase_idx_path = os.path.join(data_root, 'split/{}'.format(split_idx), '{}.txt'.format(phase))
		phase_idx = pd.read_table(phase_idx_path, header=None).values.squeeze()

		vuid_file = os.path.join(data_root, 'vuid_list.txt')
		vuids_all = pd.read_table(vuid_file, header=None, dtype=str)
		vuids_all.columns = ['vid', 'uid']

		target_file = os.path.join(data_root, 'target.npy')
		target_ori_all = np.load(target_file)[:, 0]

		target_log = np.log(target_ori_all)
		self.target_log_mean = target_log.mean()
		self.target_log_std = target_log.std()
		target_log_norm = (target_log - self.target_log_mean) / self.target_log_std
		self.target = target_log_norm[phase_idx]

		u_feat_file = os.path.join(data_root, 'user.npy')
		self.u_feat = np.load(u_feat_file)[phase_idx]

		self.v_feat = {}
		for modality in modalities:
			mod_file = os.path.join(data_root, "{}.npy".format(mod2feat_dict[modality]))
			self.v_feat[modality] = np.load(mod_file)[phase_idx]

		self.user_pop = self.get_mean_pop(vuids_all, target_ori_all, train_idx, phase_idx) # log norm pop

	def get_mean_pop(self, vuids, target, train_idxes, phase_idxes):
		### use train set to calculate mean pop
		### new users in test / val use zero mean pop

		# 1. calculate the mean pop of users in train set
		uids_known = vuids['uid'][train_idxes].tolist()
		target_known = target[train_idxes]
		unique_uids = list(set(uids_known))
		uids_vcount = np.zeros(len(unique_uids))
		uids_sum_pop = np.zeros(len(unique_uids))
		for idx, uid in enumerate(uids_known):
			uids_vcount[unique_uids.index(uid)] += 1
			uids_sum_pop[unique_uids.index(uid)] += target_known[idx]
		uids_mean_pop = uids_sum_pop / uids_vcount
		uids_mean_log_pop = np.log(uids_mean_pop)
		uids_mean_log_norm_pop = (uids_mean_log_pop - self.target_log_mean) / self.target_log_std # normalization
		uids_mean_log_pop_missing = np.zeros_like(uids_mean_log_pop[0])

		# 2. map to the phase set
		vids_mean_log_norm_pop = np.zeros((len(phase_idxes)))
		vids_unknown = vuids['vid'][phase_idxes].tolist()
		uids_unknown = vuids['uid'][phase_idxes].tolist()
		for idx, (vid, uid) in enumerate(zip(vids_unknown, uids_unknown)):
			if uid in unique_uids:
				vids_mean_log_norm_pop[idx] = uids_mean_log_norm_pop[unique_uids.index(uid)]
			else:
				vids_mean_log_norm_pop[idx] = uids_mean_log_pop_missing
		return vids_mean_log_norm_pop

	def __len__(self):
		return len(self.target)

	def __getitem__(self, index):
		samples = {
			'u_feat': torch.tensor([self.u_feat[index]], dtype=torch.float32),
			'v_feat': {key: torch.tensor(mod[index], dtype=torch.float32) for key, mod in self.v_feat.items()},
			'tgt_pop': torch.tensor([self.target[index]], dtype=torch.float32),
			'tgt_mean_pop': torch.tensor([self.user_pop[index]], dtype=torch.float32)
		}
		return samples


if __name__ == "__main__":
	# for test only
	modalities = ['visual', 'aural', 'textual']
	mod2feat_dict = {
		'visual': 'visual',
		'aural': 'aural',
		'textual': 'textual'
	}
	data4test = XiguaDataset('train', './data/', 10, modalities, mod2feat_dict)
	print(len(data4test)) # total sample num
	dataloader4test = DataLoader(data4test, batch_size=128, shuffle=True)
	print(len(dataloader4test)) # batch num
	for i_batch, batch_data in enumerate(dataloader4test):
		print(batch_data['u_feat'].shape)
		print(batch_data['v_feat'].keys())
		print(batch_data['v_feat']['visual'].shape)
		print(batch_data['v_feat']['aural'].shape)
		print(batch_data['v_feat']['textual'].shape)
		print(batch_data['tgt_pop'].shape)
		print(batch_data['tgt_mean_pop'].shape)
		break
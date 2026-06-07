'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
from utils.parser import *
from utils.utils import *
from evaluate import *
from data import *
from model import *
import torch
import numpy as np
import pandas as pd
import random
import logging

def predict(args):
	# seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# GPU / CPU
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args.cuda_idx) if torch.cuda.is_available() else "cpu")

	# load data
	data = XiguaDataset(phase='test',
						pop_len=args.pop_len,
						data_root=args.data_root,
						split_idx=args.split_idx,
						modalities=args.modalities,
	                    mod2feat_dict=args.mod2feat_dict,
	                    time_emb=(args.time_emb_dim is not None),
						logging=logging)

	# load model
	model = DMMVED(num_u=args.num_user,
	               u_in_size=args.user_dim,
	               u_emb_size=args.user_emb_dim,
	               t_emb_size=args.time_emb_dim,
	               dec_type=args.dec_type,
	               hid_size=args.hid_size,
	               mod_in_sizes=args.mod2dim_dict,
	               modalities=args.modalities,
	               drop_p=args.dropout)
	model = load_model(model, args.trained_model_path)
	model.to(device)

	# evaluate
	nmse, plcc, srcc, p_val = evaluate(model, data, args.test_batch_size, use_cuda, device)

	return nmse, plcc, srcc, p_val

def get_embs(args):
	# GPU / CPU
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args.cuda_idx) if torch.cuda.is_available() else "cpu")

	# load data
	data = XiguaDataset(phase='test',
						pop_len=args.pop_len,
						data_root=args.data_root,
						split_idx=args.split_idx,
						modalities=args.modalities,
	                    mod2feat_dict=args.mod2feat_dict,
	                    time_emb=(args.time_emb_dim is not None),
						logging=logging)

	# load model
	model = DMMVED(num_u=args.num_user,
	               u_in_size=args.user_dim,
	               u_emb_size=args.user_emb_dim,
	               t_emb_size=args.time_emb_dim,
	               dec_type=args.dec_type,
	               hid_size=args.hid_size,
	               mod_in_sizes=args.mod2dim_dict,
	               modalities=args.modalities,
	               drop_p=args.dropout)
	model = load_model(model, args.trained_model_path)
	model.to(device)

	model.eval()
	batch_size = args.test_batch_size
	total_sample_num = len(data)
	data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
	u_embs = np.empty([total_sample_num, args.hid_size], dtype=np.float32)
	v_embs = np.empty([total_sample_num, args.hid_size], dtype=np.float32)
	with torch.no_grad():
		for idx, batch_data in enumerate(data_loader):
			u_feat = batch_data['u_feat'].squeeze().to(device)
			v_feat = batch_data['v_feat']
			for mod_key in v_feat.keys():
				v_feat[mod_key] = v_feat[mod_key].to(device)
			u_emb, v_emb = model.get_embs(u_feat, v_feat)
			if use_cuda:
				u_emb = u_emb.cpu()
				v_emb = v_emb.cpu()
			true_batch_size = batch_data['tgt_pop'].shape[0]
			u_embs[idx*batch_size:idx*batch_size+true_batch_size] = u_emb
			v_embs[idx*batch_size:idx*batch_size+true_batch_size] = v_emb

	np.save(os.path.join(args.emb_save_path, 'u_embs.npy'), u_embs)
	np.save(os.path.join(args.emb_save_path, 'v_embs.npy'), v_embs)
	

if __name__ == "__main__":
	args = parse_args()
	args.trained_model_path = ""
	model_name = os.path.basename(args.trained_model_path)
	nmse, srcc, p_val = predict(args)
	metrics = pd.DataFrame([model_name, nmse, srcc, p_val]).transpose()
	metrics.columns = ['model_info',
	                   'nmse',
	                   'srcc',
	                   'pval']
	metrics.to_csv(os.path.join(args.trained_model_path, 'test_results.csv'), mode='a', sep='\t', index=False)
	
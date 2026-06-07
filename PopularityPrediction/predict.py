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
	                    data_root=args.data_root,
	                    split_idx=args.split_idx,
	                    modalities=args.modalities,
						mod2feat_dict=args.mod2feat_dict,
	                    logging=logging)

	# load model
	model = DMMVED(num_u=args.num_user,
	               u_in_size=args.user_dim,
	               u_emb_size=args.user_emb_dim,
	               dec_type=args.dec_type,
	               hid_size=args.hid_size,
	               mod_in_sizes=args.mod2dim_dict,
	               modalities=args.modalities,
	               drop_p=args.dropout)
	trained_model = load_model(model, args.trained_model_path)
	trained_model.to(device)

	# evaluate
	nmse, srcc, p_val = evaluate(trained_model, data, args.test_batch_size, use_cuda, device)

	return nmse, srcc, p_val


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
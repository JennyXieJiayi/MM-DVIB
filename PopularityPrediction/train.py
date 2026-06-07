'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
from utils.parser import *
from utils.log_helper import *
from utils.utils import *
from utils.optimizer import *
from evaluate import evaluate
from model import *
from data import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, random, logging, time


def lr_lambda_func(epoch, epoch_num):
	schedule = [[0           , 1.],
                [epoch_num/2 , 5e-1],
                [epoch_num   , 1e-1]]
	for (l_t, l), (r_t, r) in zip(schedule[:-1], schedule[1:]):
		if l_t <= epoch and epoch < r_t:
			alpha = float(epoch - l_t) / (r_t - l_t)
			return l + alpha * (r - l)
	return 1e-1


def train(args):
	# seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# log
	log_save_id = create_log_id(args.save_dir)
	logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
	logging.info(args)

	# GPU / CPU
	if args.use_cuda:
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:{}".format(args.cuda_idx) if torch.cuda.is_available() else "cpu")
	else:
		use_cuda = False
		device = torch.device('cpu')

	# load data
	data = XiguaDataset(phase='train',
						data_root=args.data_root,
						split_idx=args.split_idx,
						modalities=args.modalities,
	                    mod2feat_dict=args.mod2feat_dict,
						logging=logging)
	data_loader = DataLoader(data,
							 batch_size=args.train_batch_size,
							 shuffle=True)
	batch_num = len(data_loader)

	# construct model
	model = DMMVED(num_u=args.num_user,
	               u_in_size=args.user_dim,
	               u_emb_size=args.user_emb_dim,
	               dec_type=args.dec_type,
	               hid_size=args.hid_size,
	               mod_in_sizes=args.mod2dim_dict,
	               modalities=args.modalities,
	               drop_p=args.dropout)
	model.to(device)
	logging.info(model)

	if os.path.isfile(args.pre_model_path):
		model = load_model(model, args.pre_model_path)
		logging.info("Pre-trained model: {}".format(args.pre_model_path))
	else:
		logging.info("Parameters init...")
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	# optimizer
	optimizer = NoamOpt(args.opt_model_size,
	                    args.opt_factor,
	                    args.opt_warmup,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-3))
	logging.info(optimizer)

	# initialize metrics
	init_metrics = pd.DataFrame([['epoch_idx'], ['nMSE'], ['PLCC'], ['SRCC'], ['SRCC_p_val']]).transpose()
	init_metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t',
						index=False)
	last_save_epoch = -1
	best_epoch_nmse = -1
	nmse_min = np.inf
	plcc_cor_nmse = -np.inf
	srcc_cor_nmse = -np.inf
	best_epoch_plcc = -1
	nmse_cor_plcc = np.inf
	srcc_cor_plcc = -np.inf
	plcc_max = -np.inf

	# train model
	writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
	start_epoch_idx = args.start_epoch_idx or 1
	for epoch in range(start_epoch_idx, args.epoch_num + start_epoch_idx):
		time1 = time.time()
		model.train()
		total_loss = 0
		for idx, batch_data in enumerate(data_loader):
			batch_idx = idx + 1
			time2 = time.time()
			u_feat = batch_data['u_feat'].squeeze().to(device)
			v_feat = batch_data['v_feat']
			for mod_key in v_feat.keys():
				v_feat[mod_key] = v_feat[mod_key].to(device)
			tgt_pop = batch_data['tgt_pop'].squeeze().to(device)
			tgt_mean_pop = batch_data['tgt_mean_pop'].squeeze().to(device)
			out_mean_pop, out_pop = model.forward(u_feat, v_feat)
			batch_loss, kld_mean, recon_mean, kld_final, recon_final = model.loss(out_mean_pop, out_pop, tgt_mean_pop, tgt_pop, args.lambda_u, args.lambda_v, args.factor_uv)
			optimizer.optimizer.zero_grad() # NoamOpt
			batch_loss.backward()
			cur_lr = optimizer.step() # NoamOpt
			total_loss += batch_loss.item()
			# log to tensorboard

			if (batch_idx % args.print_every) == 0:
				logging.info(
					'Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s '
					'| Current lr {:.8f} | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch,
																		batch_idx,
																		batch_num,
																		time.time() - time2,
                                                                        cur_lr,
																		batch_loss.item(),
																		total_loss / batch_idx))
				writer.add_scalars('training_loss',
				                   {
					                   'batch_loss': batch_loss,
					                   'kld_mean': kld_mean,
					                   'recon_mean': recon_mean,
					                   'kld_final': kld_final,
					                   'recon_final': recon_final
				                   },
				                   epoch * batch_num + idx)
				writer.flush()
		logging.info(
					'Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s '
					'| Iter Mean Loss {:.4f}'.format(epoch,
													 batch_num,
													 time.time() - time1,
													 total_loss / batch_num))


		if (epoch % args.save_every) == 0:
			save_model(model, args.save_dir, epoch)
			last_save_epoch = epoch

		if (epoch % args.evaluate_every) == 0:
			time3 = time.time()
			val_data = XiguaDataset(phase='val',
									data_root=args.data_root,
									split_idx=args.split_idx,
									modalities=args.modalities,
									mod2feat_dict=args.mod2feat_dict,
			                        logging=logging)
			nmse, plcc, srcc, p_val = evaluate(model, val_data, args.test_batch_size, use_cuda, device)
			logging.info(
				'Evaluation (K={}): Epoch {:04d} | Total Time {:.1f}s '
				'| nMSE {:.4f} PLCC {:.4f} SRCC {:.4f} p_val {:.4f}'.format(5,
																epoch,
																time.time() - time3,
																nmse,
                                                                plcc,
																srcc,
																p_val))

			# save the best result
			if nmse < nmse_min:
				nmse_min = nmse
				srcc_cor_nmse = srcc
				plcc_cor_nmse = plcc
				save_model(model, args.save_dir, epoch, best_epoch_nmse, 'nmse')
				best_epoch_nmse = epoch
			if plcc > plcc_max:
				plcc_max = plcc
				nmse_cor_plcc = nmse
				srcc_cor_plcc = srcc
				save_model(model, args.save_dir, epoch, best_epoch_plcc, 'plcc')
				best_epoch_plcc = epoch

			metrics = pd.DataFrame([epoch, nmse, plcc, srcc, p_val]).transpose()
			metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t',
					   index=False)

	best_metrics = pd.DataFrame([['best_nmse_{}'.format(best_epoch_nmse), nmse_min, plcc_cor_nmse, srcc_cor_nmse], ['best_plcc_{}'.format(best_epoch_plcc), nmse_cor_plcc, plcc_max, srcc_cor_plcc]])
	best_metrics.to_csv(os.path.join(args.save_dir, 'train_results.csv'), mode='a', header=False, sep='\t', index=False)

	if last_save_epoch > -1:
		return os.path.join(args.save_dir, 'model_{}.pth'.format(last_save_epoch)), last_save_epoch + 1
	else: return 0, 0


if __name__ == "__main__":
	args = parse_args()
	train(args)
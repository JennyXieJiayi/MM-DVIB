'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
import copy, math

def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class FeedForward(nn.Module):
	def __init__(self, in_size, hid_size=None, out_size=None, activation=nn.ReLU, drop_p=0.1):
		super(FeedForward, self).__init__()
		hid_size = hid_size or in_size
		out_size = out_size or in_size
		self.ffn = nn.Sequential(
			activation(),
			nn.Linear(in_size, out_size)
		)

	def forward(self, x):
		x = self.ffn(x)
		return x

class UserEncoder(nn.Module):
	def __init__(self, num_u, in_size, emb_size, hid_size, activation, drop_p=0.2):
		super(UserEncoder, self).__init__()
		self.emb_size = emb_size
		self.user_emb = nn.Embedding(num_u, emb_size)
		self.linear = nn.Linear(emb_size+in_size-1, hid_size*4)
		self.mean_std = clones(FeedForward(in_size=hid_size*4, out_size=hid_size, activation=activation, drop_p=drop_p), 2)

	def forward(self, u_in):
		u_emb = self.user_emb(u_in[:,0].long()) * math.sqrt(self.emb_size)
		u_rep = torch.cat((u_emb, u_in[:,1:]), -1)
		u_rep = self.linear(u_rep)
		u_mean = self.mean_std[0](u_rep)
		u_std = torch.exp(self.mean_std[1](u_rep).clamp_(min=-20.,max=2.))
		return u_mean, u_std


class VideoEncoder(nn.Module):
	def __init__(self, in_sizes, hid_size, modalities, activation, drop_p=0.2):
		super(VideoEncoder, self).__init__()
		self.mod_encoder = nn.ModuleDict()
		if 'visual' in modalities:
			hid_sizes = [hid_size*4, hid_size]
			self.mod_encoder['visual'] = ModalEncoder(in_sizes['visual'], hid_sizes, activation, drop_p)
		if 'aural' in modalities:
			hid_sizes = [hid_size*4, hid_size]
			self.mod_encoder['aural'] = ModalEncoder(in_sizes['aural'], hid_sizes, activation, drop_p)
		if 'textual' in modalities:
			hid_sizes = [hid_size*4, hid_size]
			self.mod_encoder['textual'] = ModalEncoder(in_sizes['textual'], hid_sizes, activation, drop_p)

	def forward(self, mods_in, sampled_z_u):
		mods_dist = []
		for mod_in_key, mod_in in mods_in.items():
			mods_dist.append(self.mod_encoder[mod_in_key](mod_in, sampled_z_u))
		mean_list, std_list = zip(*mods_dist)
		return self.poe(torch.stack(mean_list, dim=-1), torch.stack(std_list, dim=-1))

	def poe(self, mean_list, std_list):
		# mean_list: (B, hid_size, num_mod)
		# std_list: (B, hid_size, num_mod)
		rec_var_list = 1 / (std_list ** 2)
		poe_mean = torch.sum(torch.mul(mean_list, rec_var_list), dim=-1) / torch.sum(rec_var_list, dim=-1)
		poe_std = torch.sqrt(1 / torch.sum(rec_var_list, dim=-1))
		return poe_mean, poe_std


class ModalEncoder(nn.Module):
	def __init__(self, in_size, hid_sizes, activation, drop_p=0.1):
		super(ModalEncoder, self).__init__()
		self.linear = nn.Linear(in_size, hid_sizes[0])
		self.z_u_norm = nn.LayerNorm(hid_sizes[-1])
		self.linear_u = nn.Linear(hid_sizes[-1], hid_sizes[0])
		self.mean_std = clones(FeedForward(in_size=hid_sizes[0], out_size=hid_sizes[-1], activation=activation, drop_p=drop_p), 2)

	def forward(self, mod_in, sampled_z_u):
		mod_rep = self.linear(mod_in)
		mod_rep = mod_rep - self.linear_u(self.z_u_norm(sampled_z_u))
		mod_mean = self.mean_std[0](mod_rep)
		mod_std = torch.exp(self.mean_std[1](mod_rep).clamp_(min=-20.,max=2.))
		return mod_mean, mod_std


class Sampler(nn.Module):
	def __init__(self):
		super(Sampler, self).__init__()

	def forward(self, mean, std):
		sampled_z = self.reparameterize(mean, std)
		return sampled_z

	def reparameterize(self, mean, std):
		eps = Normal(torch.zeros_like(mean), torch.ones_like(std)).sample()
		return mean + std * eps


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, drop_p, max_len=100):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=drop_p)

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0., max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0., d_model, 2) *
		                     -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)].requires_grad_(False)
		return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = scores.softmax(dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class Decoder(nn.Module):
	def __init__(self, hid_size, dec_obj, dec_type='ffn', drop_p=0.2):
		super(Decoder, self).__init__()
		self.dec_type = dec_type
		if dec_type == 'ffn':
			self.decoder = nn.Sequential(
				nn.Linear(hid_size, hid_size),
				nn.ReLU(),
				nn.Dropout(drop_p)
			)
		self.conv1d_dec = nn.Conv1d(hid_size, 1, kernel_size=1)
		self.linear_dec = nn.Linear(hid_size, 1)
		self.dec_obj = dec_obj
		if dec_obj == 'video':
			self.uv_param = nn.Parameter(torch.tensor(0.5))

	def forward(self, sampled_zs):
		# (sampled_z_v, sampled_z_u) for video and sampled_z_u for user
		if self.dec_obj == 'video':
			sampled_z = self.uv_param * sampled_zs[0] + (1 - self.uv_param) * sampled_zs[1]
		else: 
			sampled_z = sampled_zs
		if self.dec_type == 'ffn':
			out = self.decoder(sampled_z)
		return self.linear_dec(out).squeeze()
	

class DMMVED(nn.Module):
	def __init__(self, num_u, u_in_size, u_emb_size, dec_type, hid_size, mod_in_sizes, modalities, activation=nn.ReLU, drop_p=0.2):
		super(DMMVED, self).__init__()
		assert dec_type in ['ffn']
		self.u_encoder = UserEncoder(num_u, u_in_size, u_emb_size, hid_size, activation, drop_p)
		self.u_sampler = Sampler()
		self.u_decoder = Decoder(hid_size, 'user', dec_type, drop_p=drop_p)

		self.v_encoder = VideoEncoder(mod_in_sizes, hid_size, modalities, activation, drop_p)
		self.v_sampler = Sampler()
		self.decoder = Decoder(hid_size, 'video', dec_type, drop_p=drop_p)

	def forward(self, u_feat, v_feat):
		# u_feat: (B, 7) id + 6d feat
		# v_feat: dict
		#         key: visual, aural, textual
		#         value: (B, 2048), (B, 128), (B, 101)
		u_mean, u_std = self.u_encoder(u_feat)
		sampled_z_u = self.u_sampler(u_mean, u_std)
		out_mean_pop = self.u_decoder(sampled_z_u)
		v_mean, v_std = self.v_encoder(v_feat, sampled_z_u)
		sampled_z_v = self.v_sampler(v_mean, v_std)
		out_pop = self.decoder((sampled_z_v, sampled_z_u))
		self.u_mean, self.u_std, self.v_mean, self.v_std = u_mean, u_std, v_mean, v_std
		return out_mean_pop, out_pop

	def predict(self, u_feat, v_feat):
		u_mean, _ = self.u_encoder(u_feat)
		v_mean, _ = self.v_encoder(v_feat, u_mean)
		out_pop = self.decoder((v_mean, u_mean))
		return out_pop

	def loss(self, out_mean_pop, out_pop, tgt_mean_pop, tgt_pop, lambd_u, lambd_v, factor_uv):
		mu_u, std_u, mu_v, std_v = self.u_mean, self.u_std, self.v_mean, self.v_std
		recon_mean = F.mse_loss(out_mean_pop, tgt_mean_pop)
		z_u_prior = Normal(torch.zeros_like(mu_u), torch.ones_like(std_u))
		kld_mean = torch.mean(kl_divergence(Normal(mu_u, std_u), z_u_prior))
		recon_final = F.mse_loss(out_pop, tgt_pop)
		z_v_prior = Normal(torch.zeros_like(mu_v), torch.ones_like(std_v))
		kld_final = torch.mean(kl_divergence(Normal(mu_v, std_v), z_v_prior))
		loss = factor_uv * (recon_mean + lambd_u * kld_mean) + recon_final + lambd_v * kld_final
		return loss, kld_mean, recon_mean, kld_final, recon_final


if __name__ == "__main__":
	pass
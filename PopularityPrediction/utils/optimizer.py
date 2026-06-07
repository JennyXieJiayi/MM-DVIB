'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import matplotlib.pyplot as plt
import numpy as np


class NoamOpt:
	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0

	def step(self):
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()
		return rate

	def rate(self, step=None):
		if step is None:
			step = self._step
		return self.factor * \
		       (self.model_size ** (-0.5) *
		        min(step ** (-0.5), step * self.warmup ** (-1.5)))


if __name__ == "__main__":
	# Three settings of the lrate hyperparameters.
	opts = [
		NoamOpt(16, 1, 800, None),
		NoamOpt(16, 1, 400, None),
		NoamOpt(16, 1, 200, None),
		NoamOpt(64, 1, 800, None),
		NoamOpt(64, 1, 400, None),
		NoamOpt(64, 1, 200, None),
	]
	plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
	plt.legend([i for i in range(len(opts))])
	plt.show()
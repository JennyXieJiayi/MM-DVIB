'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import math


class WarmupCosine:
    def __init__(self, optimizer, total_steps, peak_lr, warmup_ratio, min_lr_ratio):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * min_lr_ratio
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.steps <= self.warmup_steps:
            rate = self.peak_lr * self.steps / self.warmup_steps
        else:
            progress = (self.steps - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            rate = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        self.optimizer.step()
        return rate

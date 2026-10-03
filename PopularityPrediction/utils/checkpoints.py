'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of MM-DVIB model in:
Disentangling User Influence and Multimodal Content for Micro-video Popularity Prediction
'''
import numpy as np
from scipy.stats import rankdata


class BestCheckpoint:
    def __init__(self, joint):
        self.joint = joint
        self.history = []
        self.candidates = {}

    def update(self, model, epoch, metrics):
        corr = metrics['srcc' if self.joint else 'plcc']
        nmse = metrics['nmse']
        if not np.isfinite([nmse, corr]).all():
            raise ValueError('Non-finite validation metrics cannot select a checkpoint')
        row = dict(epoch=epoch, nmse=nmse, correlation=corr, metrics=metrics)
        self.history.append(row)
        if not self.joint:
            if self.candidates and corr <= next(iter(self.candidates.values()))['correlation']:
                return
            self.candidates.clear()
        else:
            # Discard dominated weights; keep every epoch's metrics for ranking.
            if any(r['nmse'] <= nmse and r['correlation'] >= corr for r in self.candidates.values()):
                return
            self.candidates = {e: r for e, r in self.candidates.items()
                               if not (nmse <= r['nmse'] and corr >= r['correlation'])}
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        self.candidates[epoch] = dict(row, state=state)

    def best(self):
        if self.joint:
            ranks = rankdata([r['nmse'] for r in self.history]) + rankdata([-r['correlation'] for r in self.history])
            index = min(range(len(self.history)), key=lambda i: (ranks[i], -self.history[i]['correlation'],
                                                                 self.history[i]['nmse'], self.history[i]['epoch']))
            return self.candidates[self.history[index]['epoch']]
        return next(iter(self.candidates.values()))

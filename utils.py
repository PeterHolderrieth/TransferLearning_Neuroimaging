import torch 
#A class which tracks averages and values over time:
class AverageMeter(object):
    def __init__(self,len_rvg=None):
        self.len_rvg=len_rvg
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.len_rvg is not None:
            self.run_vec=[]
            self.run_avg=0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.len_rvg is not None:
            self.run_vec.append(val)
            n_rel=len(self.run_vec)
            if n_rel>self.len_rvg:
                self.run_vec=self.run_vec[(n_rel-self.len_rvg):]
            self.run_avg=torch.mean(torch.tensor(self.run_vec))
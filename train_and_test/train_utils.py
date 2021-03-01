import torch 
import torch.nn as nn
import numpy as np 
#A class which tracks averages and values over time:
class AverageMeter(object):
    def __init__(self,len_rvg=None,track=True):
        self.len_rvg=len_rvg
        self.track=track
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.track: 
            self.vec=[]
        if self.len_rvg is not None:
            self.run_vec=[]
            self.run_avg=0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.track: 
            self.vec.append(val)
        if self.len_rvg is not None:
            self.run_vec.append(val)
            n_rel=len(self.run_vec)
            if n_rel>self.len_rvg:
                self.run_vec=self.run_vec[(n_rel-self.len_rvg):]
            self.run_avg=torch.mean(torch.tensor(self.run_vec))

class TrainMeter(object):
    def __init__(self,len_rvg=None,track=True):
        self.tr_loss=AverageMeter(len_rvg,track)
        self.tr_eval=AverageMeter(len_rvg,track)
        self.val_loss=AverageMeter(len_rvg,track)
        self.val_eval=AverageMeter(len_rvg,track)

    def update(self,tr_loss_it=None,tr_eval_it=None,val_loss_it=None,val_eval_it=None,n=1):
        if tr_loss_it is not None: self.tr_loss.update(tr_loss_it,n)     
        if tr_eval_it is not None: self.tr_eval.update(tr_eval_it,n)      
        if val_loss_it is not None: self.val_loss.update(val_loss_it,n)        
        if val_eval_it is not None: self.val_eval.update(val_eval_it,n)

class TestMeter(object):
    def __init__(self,len_rvg=None,track=True):
        self.test_loss=AverageMeter(len_rvg,track)
        self.test_eval=AverageMeter(len_rvg,track)

    def update(self,test_loss_it=None,test_eval_it=None,n=1):   
        if test_loss_it is not None: self.test_loss.update(test_loss_it,n)        
        if test_eval_it is not None: self.test_eval.update(test_eval_it,n)

def give_optimizer(optim_type,model,lr,weight_decay,momentum):
    if optim_type=='SGD':
        return torch.optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay,momentum=momentum)
    else: 
        sys.exit("Unknown optimizer type: either 'SGD' or nothing.")

def give_lr_scheduler(scheduler_type,optimizer,epoch_dec,gamma_dec,treshold=None):
    
    if scheduler_type=='step': 
        print("Epoch decay: ", epoch_dec)
        print("Gamma decay: ", gamma_dec)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_dec, gamma=gamma_dec)
    elif scheduler_type=='plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        patience=epoch_dec, 
                                                        factor=gamma_dec,
                                                        threshold=threshold) 
    else: 
        sys.exit("Unknown scheduler type.")

def print_sep_line():
    sep_line=("--------------------------------------------------------------------------------"+
                "--------------------------------------------------------------------------------")
    print(sep_line)


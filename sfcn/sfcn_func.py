import torch 
import dp_utils as dpu
import dp_loss as dpl
#Modules:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
import sys 
import json

#Own files:
from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
from data.oasis.load_oasis3 import give_oasis_data
from epoch import go_one_epoch
from training import train
from utils import TrainMeter
from utils import print_sep_line


def give_optimizer(optim_type,model,lr,weight_decay,momentum):
    if optim_type=='SGD':
        return torch.optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay,momentum=momentum)
    else: 
        sys.exit("Unknown optimizer type: either 'SGD' or nothing.")

def give_lr_scheduler(scheduler_type,optimizer,epoch_dec,gamma_dec,treshold=None):
    
    if scheduler_type=='step': 
        return torch.optim.StepLR(optimizer, step_size=epoch_dec, gamma=gamma_dec)

    elif scheduler_type='plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        patience=epoch_dec, 
                                                        factor=gamma_dec,
                                                        threshold=threshold) 
    else: 
        sys.exit("Unknown scheduler type.")

def give_metrics(bin_min:int,bin_max:int,space:str,loss_met:str, eval_met: str, bin_step=1,sigma=1):
    if space=='continuous':
        label_translater,bin_centers=dpu.give_label_translater({
                                'type': 'label_to_bindist', 
                                'bin_step': 1,
                                'bin_range': [bin_min,bin_max],
                                'sigma': sigma})
        loss_func=dpl.give_my_loss_func({'type': loss_met,'bin_centers': bin_centers})
        eval_func=dpl.give_my_loss_func({'type': eval_met,'bin_centers': bin_centers,'probs': False})

    elif space=='binary':
        label_translater=dpu.give_label_translater({
                                'type': 'one_hot', 
                                'n_classes': 2})
        loss_func=dpl.give_my_loss_func({'type': ARGS['LOSS']})
        eval_func=dpl.give_my_loss_func({'type':'acc','thresh':0.5})
    else: 
        sys.exit("Unknown tasks.")

    return(loss_func,eval_func,label_translater)



def give_fresh_sfcn(bin_min,bin_max,dropout):
    model = SFCN(output_dim=bin_max-bin_min,dropout=dropout)
    model=nn.DataParallel(model)
    return(model)


def give_pretrained_sfcn(run:str, task: str):
    with open("hps/pretrained_sfcns.json", "r") as read_file:
        config = json.load(read_file)
    
    model_info=config[task][run]
    path_to_pretrained=osp.join(config['path'],task,model_info['file'])

    #Initialize model:
    model=SFCN(channel_number=model_info['channels'],output_dim=model_info['output_dim'])
    model=nn.DataParallel(model)

    #Load pre-trained model:
    state_dict=torch.load(path_to_pretrained)
    model.load_state_dict(state_dict)

    return(model)


def change_output_dim_sfcn(model,new_output_dim):

    c_in = model.module.classifier.conv_6.in_channels
    conv_last = nn.Conv3d(c_in, new_output_dim, kernel_size=1)
    model.module.classifier.conv_6 = conv_last

    return(model)


def train_full_model(model,....): 
    model.module.train_full_model()
    info_start="Full model is being trained."
    info_end="Full model was being trained."
    train_model(model,...,info_start=info_start,info_end=info_end)
    
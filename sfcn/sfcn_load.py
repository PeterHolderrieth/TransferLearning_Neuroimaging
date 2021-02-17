#Modules:
import torch
import torch.nn as nn
import sys 
import json
import os.path as osp 

#Own files:
from sfcn.sfcn_model import SFCN


def give_fresh_sfcn(bin_min: int, bin_max: int, dropout: float):
    '''
    bin_min,bin_max - describe the range of ages for the model ([bin_min,bin_min+1] is the most left
    bin,[bin_max-1,bin_max] the most right)
    dropout - with range [0,1] - describes the probability of nodes dropping out in the final layer.
    '''
    model = SFCN(output_dim=bin_max-bin_min,dropout=dropout)
    return nn.DataParallel(model)


def give_pretrained_sfcn(run: str, task: str):
    '''

    '''
    with open("../hps/pretrained_sfcns.json", "r") as read_file:
        config = json.load(read_file)
    
    model_info=config[task][run]
    path_to_pretrained=osp.join(config['path'],task,model_info['file'])

    #Initialize model:
    module=SFCN(channel_number=model_info['channels'],output_dim=model_info['output_dim'])
    model=nn.DataParallel(module)

    #Load pre-trained model:
    state_dict=torch.load(path_to_pretrained)
    model.load_state_dict(state_dict)

    return(model)
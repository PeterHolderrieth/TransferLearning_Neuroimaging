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

def train_full_model(model,....): 
    model.module.train_full_model()
    info_start="Full model is being trained."
    info_end="Full model is being trained."
    train_model(model,...,info_start=info_start,info_end=info_end)

def train_model(model,n_epochs,loss_func,eval_func,info_start=None,info_end=None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs: ", torch.cuda.device_count())
    
    print()
    print(info)
    print("Start: ", datetime.datetime.today())
    print_sep_line()

    meter=train(model,n_epochs,loss_func,device,train_loader,
                    val_loader,optimizer,scheduler,label_translater,eval_func)
    
    print_sep_line() 
    print("Finished: ", datetime.datetime.today())
    print()

    loss_arr=np.array(meter.tr_loss.vec)
    eval_arr=np.array(meter.tr_eval.vec)
    corr=np.corrcoef(loss_arr,eval_arr)[0,1]
    
    print("Correlation between train loss and evaluation:", "%.3f"%corr)
    print() 
    print_sep_line()
    
    return(meter)



#Set batch size and number of workers:
SHUFFLE=True
AGE_RANGE=[40,96] #Age range of data



if ARGS['PRE']=='age':
    RUNS=['run_20191206_00_epoch_best_mae.p', 
            'run_20191206_01_epoch_best_mae.p', 
            'run_20191206_02_epoch_best_mae.p', 
            'run_20191206_03_epoch_best_mae.p',
            'run_20190719_00_epoch_best_mae.p']
elif ARGS['PRE']=='sex':
    RUNS=['run_20191008_00_epoch_last.p']
else: 
    sys.exit("Unknown pre-trained type.")

RUN_NAME=RUNS[ARGS['RUN']]

print('Pre-trained on: ', ARGS['PRE'],' prediction.')
print('Run: ', f'{RUN_NAME}')

PATH_TO_PRETRAINED='/well/win-fmrib-analysis/users/lhw539/pre_trained_models/'+ARGS['PRE']+f'/{RUN_NAME}'




#Load OASIS data:
_,train_loader=give_oasis_data('train', batch_size=ARGS['BATCH_SIZE'],
                                        num_workers=ARGS['NUM_WORKERS'],
                                        shuffle=SHUFFLE,
                                        debug=debug,
                                        preprocessing=ARGS['PRO'],
                                        task=ARGS['TASK'])
                                        
_,val_loader=give_oasis_data('val', batch_size=ARGS['BATCH_SIZE'],
                                    num_workers=ARGS['NUM_WORKERS'],
                                    shuffle=SHUFFLE,
                                    debug=debug,
                                    preprocessing=ARGS['PRO'],
                                    task=ARGS['TASK'])                   


if ARGS['TRAIN']=='fresh':
    #Initialize model from scratch:
    model = SFCN(output_dim=BIN_RANGE[1]-BIN_RANGE[0],dropout=ARGS['DROPOUT'])
    model=nn.DataParallel(model)

elif ARGS['TRAIN']=='pre_full' or ARGS['TRAIN']=='pre_step':
    #Load the model:
    if ARGS['PRE']=='sex':
        model = SFCN(channel_number=[28, 58, 128, 256, 256, 64],output_dim=2)

    elif ARGS['RUN']==4:
        model = SFCN()
    else:
        model=SFCN(channel_number=[32, 64, 64, 64, 64, 64])

    model=nn.DataParallel(model)
    state_dict=torch.load(PATH_TO_PRETRAINED)
    model.load_state_dict(state_dict)
    
    #Reinitialize final layers while preserve scaling:
    model.module.reinit_final_layers_pres_scale(ARGS['INIT'])
    
    if ARGS['PRE']!='sex' or ARGS['TASK']!='sex':
        #Reshape and reinitialize the final layer:
        c_in = model.module.classifier.conv_6.in_channels
        conv_last = nn.Conv3d(c_in, BIN_RANGE[1]-BIN_RANGE[0], kernel_size=1)
        model.module.classifier.conv_6 = conv_last
    
    model.module.classifier.dropout.p=ARGS['DROP']
else: 
    sys.exit("Training mode unknown.")

#Send model to device:
model=model.to(DEVICE)

#-------------------------------------
#Training of last layer:
#-------------------------------------
if ARGS['TRAIN']=='pre_step':
    model.module.train_only_final_layers(ARGS['RETRAIN'])
    optimizer=torch.optim.SGD(model.parameters(),lr=ARGS['LR_LL'],weight_decay=ARGS['WDEC_LL'],momentum=ARGS['MOM_LL'])

    #The following learning rate scheduler decays the learning by gamma every step_size epochs:
    if ARGS['PL_LL']=='pl':
        threshold=1e-4
    elif ARGS['PL_LL']=='none': 
        #Make every change insignificant such that deterministic decay after step_size steps
        threshold=1 
    else: 
        sys.exit("Plateau or not? Either pl or none.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      patience=ARGS['PAT_LL'], 
                                                      factor=ARGS['GAMMA_LL'],
                                                      threshold=threshold) 
    print()
    print("Start training of last layer: ", datetime.datetime.today())
    print_sep_line()
    print("Number of epochs of pre-training:", ARGS['N_EPOCHS_LL'])
    meter=train(model,ARGS['N_EPOCHS_LL'],LOSS_FUNC,DEVICE,train_loader,val_loader,optimizer,scheduler,label_translater,EVAL_FUNC)
    print_sep_line()
    print("Finished training of full model.")
    print(datetime.datetime.today())

    loss_arr=np.array(meter.tr_loss.vec)
    mae_arr=np.array(meter.tr_eval.vec)
    corr=np.corrcoef(loss_arr,mae_arr)[0,1]
    print("Correlation between train loss and MAE:", "%.5f"%corr)
    print() 


#-------------------------------------
#Training of full model:
#-------------------------------------
model.module.train_full_model()
optimizer=torch.optim.SGD(model.parameters(),lr=ARGS['LR'],weight_decay=ARGS['WDEC'],momentum=ARGS['MOM'])

#The following learning rate scheduler decays the learning by gamma every step_size epochs:
if ARGS['PL']=='pl':
    threshold=1e-4
elif ARGS['PL']=='none': 
    #Make every change insignificant such that deterministic decay after step_size steps
    threshold=1 
else: 
    sys.exit("Plateau or not? Either pl or none.")
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      patience=ARGS['PAT'], 
                                                      factor=ARGS['GAMMA'],
                                                      threshold=threshold) 

print()
print("Start training of full model: ", datetime.datetime.today())
print_sep_line()
meter=train(model,ARGS['N_EPOCHS'],LOSS_FUNC,DEVICE,train_loader,val_loader,optimizer,scheduler,label_translater,EVAL_FUNC)
print_sep_line() 
print("Finished training of full model.")
print(datetime.datetime.today())

loss_arr=np.array(meter.tr_loss.vec)
eval_arr=np.array(meter.tr_eval.vec)
corr=np.corrcoef(loss_arr,eval_arr)[0,1]
print("Correlation between train loss and evaluation:", "%.3f"%corr)
print() 

'''
Utilities for later us:
writer.add_graph(model, input_data)
writer.close()
model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
print(torch.cuda.device_count()) 
'''

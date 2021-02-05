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
#Initialize tensorboard writer:
#writer = SummaryWriter('results/test/test_tb')

#Set device type:
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of GPUs: ", torch.cuda.device_count())


# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(
    BATCH_SIZE=2,
    LR=1e-2, 
    NUM_WORKERS=4,
    #DEBUG='debug',
    PRINT_EVERY=1,
    GAMMA=0.1,
    N_EPOCHS=3,
    PAT=1,
    PL='none',
    LOSS='kl',
    DROP='drop',
    PRE='full',
    WDEC=0.,
    MOM=0.0,
    PL_LL='none',
    WDEC_LL=0.0,
    MOM_LL=0.0,
    GAMMA_LL=0.1,
    N_EPOCHS_LL=3,
    PAT_LL=1,
    TRAIN='pre_step',
    LR_LL=1e-2,
    RUN=4
    )

#Debugging? Then use small data set:
ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="Debug or not.")

#Arguments for training:
ap.add_argument("-train", "--TRAIN", type=str, required=False,help="Train mode. Either 'fresh' (for fresh initialization), \
                                                                    'pre_step' (for using pre-training models and re-training of final layer) \
                                                                        or 'pre_full' for re-training of full model.")

ap.add_argument("-batch", "--BATCH_SIZE", type=int, required=False,help="Batch size.")
ap.add_argument("-n_work", "--NUM_WORKERS", type=int, required=False,help="Number of workers.")
ap.add_argument("-loss", "--LOSS", type=str, required=False,help="Loss function to use: mae or kl.")
ap.add_argument("-drop", "--DROP", type=str, required=False,help="drop for dropout and none for no dropout.")
ap.add_argument("-run", "--RUN", type=int, required=False,help="Choose pre-trained model. Either 0,1,2,3 or 4.")
ap.add_argument("-path", "--PATH", type=str, required=False,help="Path to (for later usage).")

ap.add_argument("-pl", "--PL", type=str, required=False,help="pl indicate whether we use an adaptive learning changing when loss reaches plateu (True) or none for deterministic decay.")
ap.add_argument("-pre", "--PRE", type=str, required=False,help="Preprocessing. Either full, min or none.")
ap.add_argument("-wdec", "--WDEC", type=float, required=False,help="Weight decay.")
ap.add_argument("-mom", "--MOM", type=float, required=False,help="Momentum for SGD.")
ap.add_argument("-gamma", "--GAMMA", type=float, required=False,help="Decay factor for learning rate schedule.")
ap.add_argument("-epochs", "--N_EPOCHS", type=int, required=False,help="Number of epochs.")
ap.add_argument("-pat", "--PAT", type=int, required=False,help="Patience, i.e. number of steps until lr is diminished.")
ap.add_argument("-lr", "--LR", type=float, required=False, help="Learning rate.")

ap.add_argument("-pl_ll", "--PL_LL", type=str, required=False,help="pl for last layer training.")
ap.add_argument("-wdec_ll", "--WDEC_LL", type=float, required=False,help="Weight decay for last layer training.")
ap.add_argument("-mom_ll", "--MOM_LL", type=float, required=False,help="Momentum last layer training.")
ap.add_argument("-gamma_ll", "--GAMMA_LL", type=float, required=False,help="Decay factor last layer training")
ap.add_argument("-epochs_ll", "--N_EPOCHS_LL", type=int, required=False,help="Number of epochs for last layer training.")
ap.add_argument("-pat_ll", "--PAT_LL", type=int, required=False,help="Patience for last layer training")
ap.add_argument("-lr_ll", "--LR_LL", type=float, required=False, help="Learning rate for last layer training.")

#ap.add_argument("-seed","--SEED", type=int, required=False, help="Seed for randomness.")

#Arguments for tracking:
ARGS = vars(ap.parse_args())
print(ARGS)
#Set batch size and number of workers:
SHUFFLE=True
AGE_RANGE=[40,96] #Age range of data
if ARGS['task']=='age':
    BIN_RANGE=[37,99] #Enlarge age range with 3 at both sides to account for border effecs
elif ARGS['task']=='sex':
    BIN_RANGE=[0,2]
else: 
    sys.exit("Unknown tasks.")
n_bins=BIN_RANGE[1]-BIN_RANGE[0]
BIN_STEP=1
SIGMA=1
print("Debug: ", ARGS['DEBUG'])
print("Patience: ", ARGS['PAT_LL'])


RUNS=['run_20191206_00', 'run_20191206_01', 'run_20191206_02', 'run_20191206_03','run_20190719_00']
RUN_NAME=RUNS[ARGS['RUN']]
print('Run: ', f'{RUN_NAME}')
PATH_TO_PRETRAINED= f'/well/win-fmrib-analysis/users/lhw539/pre_trained_models/{RUN_NAME}_epoch_best_mae.p'

sep_line=("---------------------------------------------------------------------------------------------------"+
    "-------------------")

if ARGS['DEBUG']=='debug':
    debug=True
elif ARGS['DEBUG']=='full':
    debug=False 
else: 
    sys.exit("Unvalid debug flag.")

#Load OASIS data:
_,train_loader=give_oasis_data('train', batch_size=ARGS['BATCH_SIZE'],
                                        num_workers=ARGS['NUM_WORKERS'],
                                        shuffle=SHUFFLE,
                                        debug=debug,
                                        preprocessing=ARGS['PRE'],
                                        task=ARGS['task'])
                                        
_,val_loader=give_oasis_data('val', batch_size=ARGS['BATCH_SIZE'],
                                    num_workers=ARGS['NUM_WORKERS'],
                                    shuffle=SHUFFLE,
                                    debug=debug,
                                    preprocessing=ARGS['PRE'],
                                    task=ARGS['task'])                   

#Set the label translater:
label_translater=dpu.give_label_translater({
                            'type': 'label_to_bindist', 
                            'bin_step': BIN_STEP,
                            'bin_range': BIN_RANGE,
                            'sigma': SIGMA})
#Set the loss function:
if ARGS['LOSS']=='mae':
    LOSS_FUNC=dpl.my_MAELoss #my_KLDivLoss
elif ARGS['LOSS']=='kl':
    LOSS_FUNC=dpl.my_KLDivLoss
elif ARGS['LOSS']=='ent':
    LOSS_FUNC=dpl.my_cross_entropy
else: 
    sys.exit("Unknown loss.")

#Set evaluation function:
EVAL_FUNC=dpl.give_bin_eval(bin_centers=None)


if ARGS['DROP']=='drop':
    dropout=True
elif ARGS['DROP']=='none':
    dropout=False
else: 
    sys.exit("Dropout or not? Either drop or none.")

if ARGS['TRAIN']=='fresh':
    #Initialize model from scratch:
    model = SFCN(output_dim=BIN_RANGE[1]-BIN_RANGE[0],dropout=dropout)
    model=nn.DataParallel(model)

elif ARGS['TRAIN']=='pre_full' or ARGS['TRAIN']=='pre_step':
    #Load the model:
    if ARGS['RUN']==4:
        model = SFCN()
    else:
        model=SFCN(channel_number=[32, 64, 64, 64, 64, 64])
    model=nn.DataParallel(model)
    
    state_dict=torch.load(PATH_TO_PRETRAINED)#,map_location=DEVICE)
    model.load_state_dict(state_dict)


    #Reshape and reinitialize the final layer:
    c_in = model.module.classifier.conv_6.in_channels
    conv_last = nn.Conv3d(c_in, BIN_RANGE[1]-BIN_RANGE[0], kernel_size=1)
    model.module.classifier.conv_6 = conv_last
    if dropout is False:
        model.module.classifier.dropout.p=0.
else: 
    sys.exit("Training mode unknown.")

#Send model to device:
model=model.to(DEVICE)

#-------------------------------------
#Training of last layer:
#-------------------------------------
if ARGS['TRAIN']=='pre_step':
    model.module.train_final_layer()
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
    print(sep_line) 
    print("Number of epochs of pre-training:", ARGS['N_EPOCHS_LL'])
    meter=train(model,ARGS['N_EPOCHS_LL'],LOSS_FUNC,DEVICE,train_loader,val_loader,optimizer,scheduler,label_translater,EVAL_FUNC)
    print(sep_line)   
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
print(sep_line)
meter=train(model,ARGS['N_EPOCHS'],LOSS_FUNC,DEVICE,train_loader,val_loader,optimizer,scheduler,label_translater,EVAL_FUNC)
print(sep_line)    
print("Finished training of full model.")
print(datetime.datetime.today())

loss_arr=np.array(meter.tr_loss.vec)
mae_arr=np.array(meter.tr_eval.vec)
corr=np.corrcoef(loss_arr,mae_arr)[0,1]
print("Correlation between train loss and MAE:", "%.5f"%corr)
print() 

    


'''
Utilities for later us:
writer.add_graph(model, input_data)
writer.close()
model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
print(torch.cuda.device_count()) 
'''

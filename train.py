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
import utils

#Initialize tensorboard writer:
#writer = SummaryWriter('results/test/test_tb')

#Set device type:
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")


# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(
    BATCH_SIZE=2,
    LR=1e-2, 
    NUM_WORKERS=4,
    DEBUG='debug',
    PRINT_EVERY=1,
    GAMMA=0.1,
    N_EPOCHS=3,
    TRAIN='full',
    INIT='fresh',
    PAT=10,
    PL=False,
    LOSS='mae',
    DROP=False
    )

#Debugging? Then use small data set:
ap.add_argument("-debug", "--DEBUG", type=str, required=False,help="Debug or not.")

#Arguments for training:
ap.add_argument("-batch", "--BATCH_SIZE", type=int, required=False,help="Batch size.")
ap.add_argument("-n_work", "--NUM_WORKERS", type=int, required=False,help="Number of workers.")
ap.add_argument("-lr", "--LR", type=float, required=False, help="Learning rate.")
ap.add_argument("-gamma", "--GAMMA", type=float, required=False,help="Decay factor for learning rate schedule.")
ap.add_argument("-epochs", "--N_EPOCHS", type=int, required=False,help="Number of epochs.")
ap.add_argument("-train", "--TRAIN", type=str, required=False,help="Train mode (from scratch or pre-trained model.)")
ap.add_argument("-init", "--INIT", type=str, required=False,help="Train mode (from scratch or pre-trained model.)")
ap.add_argument("-pat", "--PAT", type=int, required=False,help="Patience, i.e. number of steps until lr is diminished.")
ap.add_argument("-pl", "--PL", type=bool, required=False,help="Bool to indicate whether we use an adaptive learning changing when loss reaches plateu (True) or just rate decay.")
ap.add_argument("-loss", "--LOSS", type=str, required=False,help="Loss function to use: mae or kl.")
ap.add_argument("-drop", "--DROP", type=bool, required=False,help="Dropout or not?")

#ap.add_argument("-seed","--SEED", type=int, required=False, help="Seed for randomness.")

#Arguments for tracking:
ARGS = vars(ap.parse_args())

#Set batch size and number of workers:
SHUFFLE=True
AGE_RANGE=[40,96] #Age range of data
BIN_RANGE=[37,99] #Enlarge age range with 3 at both sides to account for border effecs
n_bins=BIN_RANGE[1]-BIN_RANGE[0]
BIN_STEP=1
SIGMA=1
PATH_TO_PRETRAINED='pre_trained_models/brain_age/run_20190719_00_epoch_best_mae.p'

if ARGS['INIT']=='fresh':
    #Initialize model from scratch:
    model = SFCN(output_dim=BIN_RANGE[1]-BIN_RANGE[0],dropout=ARGS['DROP'])
    model=nn.DataParallel(model)

elif ARGS['INIT']=='pre':
    #Load the model:
    model = SFCN()
    model=nn.DataParallel(model)
    state_dict=torch.load(PATH_TO_PRETRAINED)#,map_location=DEVICE)
    model.load_state_dict(state_dict)

    #Reshape and reinitialize the final layer:
    c_in = model.module.classifier.conv_6.in_channels
    conv_last = nn.Conv3d(c_in, BIN_RANGE[1]-BIN_RANGE[0], kernel_size=1)
    model.module.classifier.conv_6 = conv_last
    if ARGS['DROP'] is False:
        model.module.classifier.dropout.p=0.
else: 
    sys.exit("Initialization unknown.")


#Set train mode:
if ARGS['TRAIN']=='full':
    model.module.train_full_model()
elif ARGS['TRAIN']=='last':
    model.module.train_last_layer()
#elif ARGS['TRAIN']=='none':
#    model.module.train_nothing()
else: 
    sys.exit("Which parameters to train?")

optimizer=torch.optim.SGD(model.parameters(),lr=ARGS['LR'])#,weight_decay=.1)

#The following learning rate scheduler decays the learning by gamma every step_size epochs:
if ARGS['PL']:
    threshold=1e-4
else: 
    #Make every change insignificant such that deterministic decay after step_size steps
    threshold=1 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      patience=ARGS['PAT'], 
                                                      factor=ARGS['GAMMA'],
                                                      threshold=threshold) 
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
                                        debug=debug)
_,val_loader=give_oasis_data('val', batch_size=ARGS['BATCH_SIZE'],
                                    num_workers=ARGS['NUM_WORKERS'],
                                    shuffle=SHUFFLE,
                                    debug=debug)

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
else: 
    sys.exit("Unknown loss.")

#Set evaluation function:
EVAL_FUNC=dpl.give_bin_eval(bin_centers=None)

#Set average meters:
length_avg=20
meter=utils.TrainMeter(length_avg)


print()
print("Start training.")
print("---------------------------------------------------------------------------------------------------"+
      "-------------------") 

for epoch in range(ARGS['N_EPOCHS']):
     
    #Parameters of last layer:
    #par_llayer=model.module.state_dict()['classifier.conv_6.weight'].flatten().cpu()

    #Get learning rate:    
    lr=optimizer.state_dict()['param_groups'][0]['lr']
    
    #Go one epoch:
    results_tr=go_one_epoch('train',model=model,
                                loss_func=LOSS_FUNC,
                                device=DEVICE,
                                data_loader=train_loader,
                                optimizer=optimizer,
                                label_translater=label_translater,
                                eval_func=EVAL_FUNC)
    #Go one epoch:
    results_val=go_one_epoch('test',model=model,
                                loss_func=LOSS_FUNC,
                                device=DEVICE,
                                data_loader=val_loader,
                                optimizer=optimizer,
                                label_translater=label_translater,
                                eval_func=EVAL_FUNC)
    
    #results_tr={'loss': torch.rand(1).item(),'eval': torch.rand(1).item()}
    #results_val={'loss': torch.rand(1).item(),'eval': torch.rand(1).item()}

    #Update logging:
    meter.update(tr_loss_it=results_tr['loss'],
                 tr_eval_it=results_tr['eval'],
                 val_loss_it=results_val['loss'],
                 val_eval_it=results_val['eval'])
    
    
    #Parameters new layers:
    #par_nlayer=model.module.state_dict()['classifier.conv_6.weight'].flatten().cpu()
    #abs_diff=torch.abs(par_llayer-par_nlayer)
    #print("Maximum difference: ", abs_diff.max().item())
    #print("Minimum difference: ", abs_diff.min().item())    
    #print("Mean difference: ", abs_diff.mean().item())

    #Print update:
    if epoch%ARGS['PRINT_EVERY']==0:
        print(("|epoch: %3d | lr: %.3f |"+ 
                  "train loss: %.5f |train loss ravg: %.5f |"+
                  "train MAE:  %.5f |train MAE ravg:  %.5f |"+
                  "val loss: %.5f |val loss ravg: %.5f |"+
                  "val MAE:  %.5f |val MAE ravg:  %.5f |")%(epoch,
                  lr,meter.tr_loss.val,meter.tr_loss.run_avg,meter.tr_eval.val,meter.tr_eval.run_avg,
                  meter.val_loss.val,meter.val_loss.run_avg,meter.val_eval.val,meter.val_eval.run_avg))

    scheduler.step(meter.tr_loss.val)

print("---------------------------------------------------------------------------------------------------"+
      "------------------")    
print("Finished training.")

loss_arr=np.array(meter.tr_loss.vec)
mae_arr=np.array(meter.tr_eval.vec)
corr=np.corrcoef(loss_arr,mae_arr)[0,1]
print("Correlation between train loss and MAE:", "%.5f"%corr)

'''
Utilities for later us:
writer.add_graph(model, input_data)
writer.close()
model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
print(torch.cuda.device_count()) 
'''
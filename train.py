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
    LR=1e-1, 
    NUM_WORKERS=4,
    DEBUG=True,
    PRINT_EVERY=1,
    GAMMA=1.,
    N_EPOCHS=3)

#Debugging? Then use small data set:
ap.add_argument("-debug", "--DEBUG", type=bool, required=False,help="Debug or not.")

#Arguments for training:
ap.add_argument("-batch", "--BATCH_SIZE", type=int, required=False,help="Batch size.")
ap.add_argument("-n_work", "--NUM_WORKERS", type=int, required=False,help="Number of workers.")
ap.add_argument("-lr", "--LR", type=float, required=False, help="Learning rate.")
ap.add_argument("-gamma", "--GAMMA", type=float, required=False,help="Decay factor for learning rate schedule.")
ap.add_argument("-epochs", "--N_EPOCHS", type=int, required=False,help="Number of epochs.")
#ap.add_argument("-seed","--SEED", type=int, required=False, help="Seed for randomness.")
#ap.add_argument("-continue","--CONTINUE",type=str,required=False,help="File to continue training")

#Arguments for tracking:
ARGS = vars(ap.parse_args())

#Set batch size and number of workers:
SHUFFLE=True
AGE_RANGE=[40,96] #Age range of data
BIN_RANGE=[37,99] #Enlarge age range with 3 at both sides to account for border effecs
n_bins=BIN_RANGE[1]-BIN_RANGE[0]
BIN_STEP=1
SIGMA=1
DROPOUT=False
BATCH_NORM=True
N_DECAYS=5
PATH_TO_PRETRAINED='pre_trained_moddels/brain_age/run_20190719_00_epoch_best_mae.p'

#Set initialization of weights:
INITIALIZE='pre'

if INITIALIZE=='fresh':
    #Initialize model from scratch:
    model = SFCN(output_dim=BIN_RANGE[1]-BIN_RANGE[0],dropout=DROPOUT,batch_norm=BATCH_NORM)

elif INITIALIZE=='pre':
    #Load the model:
    model = SFCN()
    model=nn.DataParallel(model)
    state_dict=torch.load(PATH_TO_PRETRAINED)#,map_location=DEVICE)
    model.load_state_dict(state_dict)

    #Reshape and reinitialize the final layer:
    c_in = model.module.classifier.conv_6.in_channels
    conv_last = nn.Conv3d(c_in, BIN_RANGE[1]-BIN_RANGE[0], kernel_size=1)
    model.module.classifier.conv_6 = conv_last
    if DROPOUT is False:
        model.module.classifier.dropout.p=0.
else: 
    sys.exit("Initialization unknown.")

#Set parameters to train:
TRAIN='none'

if TRAIN=='full':
    model.module.train_full_model()
elif TRAIN=='last':
    model.module.train_last_layer()
elif TRAIN=='none':
    model.module.train_nothing()
else: 
    sys.exit("Which parameters to train?")

'''class test_model(nn.Module):
    def __init__(self, bin_range, n_x=160,n_y=192,n_z=160)expo:
        super(test_model, self).__init__()
        self.linear=nn.Linear(n_x*n_y*n_z,BIN_RANGE[1]-BIN_RANGE[0])

    def forward(self,x):
        x=x.view(x.size(0),-1)
        out=F.log_softmax(self.linear(x))
        return(out)

model=test_model(bin_range=BIN_RANGE)
'''
#model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
#print(torch.cuda.device_count()) 
optimizer=torch.optim.SGD(model.parameters(),lr=ARGS['LR'])#,weight_decay=.1)
#The following learning rate scheduler decays the learning by gamma every step_size epochs:
step_size=max(int(np.floor(ARGS['N_EPOCHS']/N_DECAYS)),1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=ARGS['GAMMA']) #gamma=1 means no decay

#Load OASIS data:
_,train_loader=give_oasis_data('train',batch_size=ARGS['BATCH_SIZE'],num_workers=ARGS['NUM_WORKERS'],shuffle=SHUFFLE,debug=ARGS['DEBUG'])
_,val_loader=give_oasis_data('val',batch_size=ARGS['BATCH_SIZE'],num_workers=ARGS['NUM_WORKERS'],shuffle=SHUFFLE,debug=ARGS['DEBUG'])

#Set the label translater:
label_translater=dpu.give_label_translater({
                            'type': 'label_to_bindist', 
                            'bin_step': BIN_STEP,
                            'bin_range': BIN_RANGE,
                            'sigma': SIGMA})
LOSS_FUNC=dpl.my_MAELoss #my_KLDivLoss
EVAL_FUNC=dpl.give_bin_eval(bin_centers=None)

length_avg=10
loss_meter=utils.AverageMeter(length_avg)
mae_meter=utils.AverageMeter(length_avg)
loss_list=[]
mae_list=[]
print()
print("Start training.")
    

#print(model.state_dict()['classifier.conv_6.weight'].flatten())
for epoch in range(ARGS['N_EPOCHS']):
    #Parameters of last layer:
    #par_llayer=model.module.state_dict()['classifier.conv_6.weight'].flatten().cpu()

    lr=optimizer.state_dict()['param_groups'][0]['lr']
    
    results=go_one_epoch('train',model=model,
                                loss_func=LOSS_FUNC,
                                device=DEVICE,
                                data_loader=train_loader,
                                optimizer=optimizer,
                                label_translater=label_translater,
                                eval_func=EVAL_FUNC)
    
    #Update logging:
    loss_meter.update(results['loss'])
    mae_meter.update(results['eval'])
    loss_list.append(results['loss'])
    mae_list.append(results['eval'])
    
    #Parameters new layers:
    #par_nlayer=model.module.state_dict()['classifier.conv_6.weight'].flatten().cpu()
    #abs_diff=torch.abs(par_llayer-par_nlayer)
    #print("Maximum difference: ", abs_diff.max().item())
    #print("Minimum difference: ", abs_diff.min().item())    
    #print("Mean difference: ", abs_diff.mean().item())

    #Print update:
    if epoch%ARGS['PRINT_EVERY']==0:
        print("| Epoch: %3d | train loss: %.5f | train MAE:  %.5f | learning rate: %.3f |"%(epoch,loss_meter.run_avg,mae_meter.run_avg,lr))

    scheduler.step()

loss_arr=np.array(loss_list)
mae_arr=np.array(mae_list)
print("Correlation between loss and MAE:", np.corrcoef(loss_arr,mae_arr)[0,1])

'''
print(type(DEVICE))
# Example
model = SFCN()
print(type(model))
model = torch.nn.DataParallel(model)
fp_ = './pre_trained_models/brain_age/run_20190719_00_epoch_best_mae.p'
model.load_state_dict(torch.load(fp_,map_location=DEVICE))
model=model.to(DEVICE)
print(model)

# Example data: some random brain in the MNI152 1mm std space
data = np.random.rand(182, 218, 182)
label = np.array([71.3,]) # Assuming the random subject is 71.3-year-old.

# Transforming the age to soft label (probability distribution)
bin_range = [42,82]
bin_step = 1
sigma = 1
y, bc = dpu.num2vect(label, bin_range, bin_step, sigma)

y = torch.tensor(y, dtype=torch.float32)
print(f'Label shape: {y.shape}')

# Preprocessing
data = data/data.mean()
data = dpu.crop_center(data, (160, 192, 160)) 
# Move the data from numpy to torch tensor on GPU
sp = (1,1)+data.shape
data = data.reshape(sp)
input_data = torch.tensor(data, dtype=torch.float32).to(DEVICE)
print(f'Input data shape: {input_data.shape}')
print(f'dtype: {input_data.dtype}')

# Evaluation
model.eval() # Don't forget this. BatchNorm will be affected if not in eval mode.
#with torch.no_grad():
print(datetime.datetime.today())
output = model(input_data)
print(datetime.datetime.today())  

# Output, loss, visualisation
x = output[0].cpu().reshape([1, -1])
print(f'Output shape: {x.shape}')
loss = dpl.my_KLDivLoss(x, y).detach().numpy()

# Prediction, Visualisation and Summary
x = x.detach().numpy().reshape(-1)
y = y.detach().numpy().reshape(-1)

prob = np.exp(x)
pred = prob@bc #Scalar product
plt.bar(bc, prob)
plt.title(f'Prediction: age={pred:.2f}\nloss={loss}')
plt.show()

x=np.array([3,-1,2])
y=np.array([0.5,-1,7])


writer.add_graph(model, input_data)
writer.close()
'''

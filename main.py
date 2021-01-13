#Modules:
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
from torch.utils.tensorboard import SummaryWriter

#Own files:
from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
from data.oasis.load_oasis3 import give_oasis_data
from epoch import go_one_epoch

#Initialize tensorboard writer:
#writer = SummaryWriter('results/test/test_tb')

#Set device type:nv
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

#If debugging is true, we only use a small subset of the train and validation data:
DEBUG=True

#Set batch size and number of workers:
BATCH_SIZE=2
NUM_WORKERS=4
SHUFFLE=True
LR=1e-1
BIN_RANGE=[40,96]
BIN_STEP=1
SIGMA=1
N_EPOCHS=300
PRINT_EVERY=1

#Set model:
model = SFCN(output_dim=56,dropout=False)
#model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
#print(torch.cuda.device_count())
optimizer=torch.optim.SGD(model.parameters(),lr=LR)
#The following learning rate scheduler decays the learning by gamma every step_size epochs:
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=np.floor(N_EPOCHS/4), gamma=.1) #gamma=1 means no decay

#Load OASIS data:
_,train_loader=give_oasis_data('train',batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=SHUFFLE,debug=DEBUG)
_,val_loader=give_oasis_data('val',batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,shuffle=SHUFFLE,debug=DEBUG)

#Set the label translater:
label_translater=dpu.give_label_translater({ 'type': 'label_to_bindist', 
                            'bin_step': BIN_STEP,
                            'bin_range': BIN_RANGE,
                            'sigma': SIGMA})
LOSS_FUNC=dpl.my_KLDivLoss
EVAL_FUNC=dpl.give_bin_eval(bin_centers=None)

print()
print("Start training.")
for epoch in range(N_EPOCHS):
    lr=optimizer.state_dict()['param_groups'][0]['lr']
    results=go_one_epoch('train',model,LOSS_FUNC,DEVICE,train_loader,optimizer,label_translater,eval_func=EVAL_FUNC)
    if epoch%PRINT_EVERY==0:
        print("| Epoch: %3d | train loss: %.5f | train MAE:  %.5f | learning rate: %.3f |"%(epoch,results['loss'],results['eval'],lr))
    scheduler.step()

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

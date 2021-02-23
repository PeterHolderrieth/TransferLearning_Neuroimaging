import numpy as np 
import torch 
from torch.autograd import Variable
import sys 
from sfcn.sfcn_load import give_pretrained_sfcn
from data.oasis.load_oasis3 import give_oasis_data
import matplotlib.pyplot as plt 

def compute_activation(model,x, filter_index,device=None):
    '''
        model - SFCN 
        x - input MRI 
        filter_index - int 
    '''
    activation = model.module.feature_extractor(x)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, filter_index]
    return torch.mean(filter_activation)

def maximize_activation(model,x,filter_index,n_epochs,lr,device=None):
    if device is not None: 
        x=x.to(device)
    x = Variable(x, requires_grad=True) 
    model.module.train_nothing()
    model.eval()
    loss_list=[]
    for it in range(n_epochs):
        loss=compute_activation(model,x,filter_index,device)
        loss.backward()
        x.data=x.data+lr*x.grad.data
        x.grad.zero_()
        print("Loss: %.6f"%loss)
        loss_list.append(loss)
    return(x,loss_list)



_,train_loader=give_oasis_data('train', batch_size=1,
                                        num_workers=4,
                                        shuffle=True,
                                        debug='debug',
                                        preprocessing='min',
                                        task='age',
                                        share=1.)

n_it=10
model=give_pretrained_sfcn("0", "age")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of GPUs: ", torch.cuda.device_count())
model=model.to(device)
lr=100
n_epochs=1000

for it in range(n_it):
    filter_index=torch.randint(low=0,high=40,size=[]).item()
    x,y=next(iter(train_loader))
    print(x.shape)
    maximizing_image,_=maximize_activation(model,x,filter_index,n_epochs,lr,device)
    maximizing_image=maximizing_image.squeeze().cpu().detach().numpy()
    x=x.squeeze().cpu().detach().numpy()
    fig, ax=plt.subplots(ncols=2,nrows=1,figsize=(20,10))
    ind=80
    ax[0].imshow(x[ind])
    ax[1].imshow(maximizing_image[ind])
    print("Difference:", np.linalg.norm(maximizing_image-x))
    filename="Test_long_"+str(it)+"_"+str(filter_index)+".pdf"
    plt.savefig(filename)



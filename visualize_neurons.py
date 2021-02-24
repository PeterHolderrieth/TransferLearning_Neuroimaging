import numpy as np 
import torch 
from torch.autograd import Variable
import sys 
from sfcn.sfcn_load import give_pretrained_sfcn
from data.oasis.load_oasis3 import give_oasis_data
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize

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

def maximize_activation(model,x,filter_index,n_epochs,lr,alpha=1.,device=None,print_every=100):
    if device is not None: 
        x=x.to(device)
    x = Variable(x, requires_grad=True) 
    model.module.train_nothing()
    model.eval()
    loss_list=[]
    for it in range(n_epochs):
        loss=compute_activation(model,x,filter_index,device)
        loss.backward()
        x.data+=lr*x.grad.data-alpha*x.data
        x.grad.zero_()
        if it%print_every==0:
            print("Epoch: %3d || Activ: %.6f"%(it,loss))
        loss_list.append(loss)
    return(x,loss_list)

model=give_pretrained_sfcn("0", "age")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of GPUs: ", torch.cuda.device_count())
model=model.to(device)

_,train_loader=give_oasis_data('train', batch_size=1,
                                        num_workers=4,
                                        shuffle=True,
                                        debug='debug',
                                        preprocessing='min',
                                        task='age',
                                        share=1.)


n_it=100
lr_list=[0.1,1.,10.,100.,0.1]
n_epochs=50000
alpha_list=[1e-6,1e-5,1e-4,1e-3]


for permute in [True, False]:
    for it in range(n_it):
        x,y=next(iter(train_loader))
        shape=x.shape
        if permute:
            x=x.flatten()
            x=x[torch.randperm(x.shape[0])]
            x=x.reshape(shape)
        x_np=x.squeeze().cpu().detach().numpy()
        filter_index=torch.randint(low=0,high=40,size=[]).item()
        alpha_index=torch.randint(low=0,high=len(alpha_list),size=[]).item()
        lr_index=torch.randint(low=0,high=len(lr_list),size=[]).item()
        alpha=alpha_list[alpha_index]
        lr=lr_list[lr_index]
        maximizing_image,loss_list=maximize_activation(model,x,filter_index,n_epochs,lr,alpha,device)
        maximizing_image=maximizing_image.squeeze().cpu().detach().numpy()
        fig, ax=plt.subplots(ncols=4,nrows=1,figsize=(40,10))
        ind=torch.randint(low=60,high=100,size=[]).item()
        nm_x=Normalize(vmin=x_np.min(), vmax=x_np.max(), clip=True)
        ax[0].imshow(x_np[ind],'gray',norm=nm_x)
        ax[1].imshow(maximizing_image[ind],'gray')
        ax[2].imshow(maximizing_image[ind],'gray',norm=nm_x)
        ax[3].plot(loss_list)
        fig.suptitle("Learning rate: %.5f || Alpha: %.5f"%(lr,alpha))
        print("Difference:", np.linalg.norm(maximizing_image-x_np))
        filename="Max_activ_"+str(it)+"_"+str(filter_index)+"_"+str(permute)+".pdf"
        plt.savefig(filename)

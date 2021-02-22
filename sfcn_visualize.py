
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
x,y=next(iter(train_loader))
model=give_pretrained_sfcn("0", "age")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of GPUs: ", torch.cuda.device_count())
model=model.to(device)
filter_index=0
n_epochs=1000
lr=1e-1
maximizing_image,_=maximize_activation(model,x,filter_index,n_epochs,lr,device)

maximizing_image=maximizing_image.squeeze().cpu().detach().numpy()
x.squeeze()
plt.imshow(maximizing_image[80])
plt.savefig("Test_long.pdf")



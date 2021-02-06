import numpy as np
import torch
import sys
from scipy.stats import norm

def give_label_translater(kwd):

    if kwd['type']=='label_to_bindist':
        bin_length = kwd['bin_range'][1]-kwd['bin_range'][0]
        if not bin_length % kwd['bin_step'] == 0:
            sys.exit("bin's range should be divisible by bin_step!")

        bin_number = int(bin_length / kwd['bin_step'])
        bin_centers = kwd['bin_range'][0]+ float(kwd['bin_step']) / 2 + kwd['bin_step'] * torch.arange(bin_number)  
        
        def label_to_bindist(x, bin_step=kwd['bin_step'],bin_centers=bin_centers, sigma=kwd['sigma'], normalize=True):
            """
            Function to convert a numerical vector x (hard label) into a bin distribution, 
            e.g. the age 71.6 is either converted into bin "10", i.e. a deterministic distribution, (if sigma=0)
            or a "normal" distribution over age bins with mean 10 and variance sigma (if sigma>0).
            Input:
                x: float or torch.Tensor of shape (n) - numeric value which we put in a valid label
                bin_range: (start, end), size-2 tuple
                bin_step: int - should be a divisor of |end-start|
                sigma:  = 0 for 'hard label', v (see output) is integer (index of label)
                        > 0 for 'soft label', v is vector of sha(distribution with mean label)
                        < 0 error. Returns error message.      
            Output:
                v - label output - shape (n,bin_number) and (bin_number) for n=1
                bin_centers - the centers of the bins 
            """
            bin_centers_=bin_centers.to(x.device)
            #If sigma is zero, set v to be index such that bin_centers[v] is closest
            #to x:
            if sigma == 0:
                abs_diff=torch.abs(x[:,None]-bin_centers_[None,:])
                v=torch.argmin(abs_diff,dim=1)
                return v
            
            #If sigma is greater than zero, then return the probability of 
            #a bin under the normal distribution:
            elif sigma > 0:
                x1=bin_centers_ - float(bin_step) / 2 #Left bin boundary
                x2=bin_centers_ + float(bin_step) / 2 #Right bin boundary
                dist=torch.distributions.Normal(loc=0.,scale=sigma) 
                v=dist.cdf(x2[None,:]-x[:,None])-dist.cdf(x1[None,:]-x[:,None]) #Probability of bin-interval 
                #Normalize v: 
                if normalize:
                    v=v/v.sum(dim=1)[:,None]
                return v               
            else:
                sys.exit("Sigma must be >=0.")

        return (label_to_bindist,bin_centers)
   
    elif kwd['type']=='identity':

        def identity(x): 
            return (x)
    
    elif kwd['type']=='one_hot':
        def one_hot(x,n_classes=kwd['n_classes']):
            one_hot_targets = torch.eye(n_classes,device=x.device)[x.long()]
            return(one_hot_targets)

        return(one_hot)

    else: 
        sys.exit("Unkown type.")

def crop_center(data, out_sp):
    """
    Returns the center part of volume data of shape out_sp
    Input: 
        data - torch.Tensor/np.array - shape in_sp=(n1,m1,k1) or in_sp=(batch_size,n1,m1,k1)
        out_sp=(n2,m2,k2) - torch.Tensor/np.array/list of length 3 
    Output:
        data_crop - torch.Tensor/np.array - shape out_sp=(*,n2,m2,k2)
    Require: n1>=n2, m1>=m2, k1>=k2
    """
    in_sp = data.shape
    nd = np.ndim(data) #number of dimensions

    #Get the limits:
    x_crop_low = int(np.floor((in_sp[-3] - out_sp[-3]) / 2))
    x_crop_high = in_sp[-3]-int(np.ceil((in_sp[-3] - out_sp[-3]) / 2))
    y_crop_low = int(np.floor((in_sp[-2] - out_sp[-2]) / 2))
    y_crop_high = in_sp[-2]-int(np.ceil((in_sp[-2] - out_sp[-2]) / 2))
    z_crop_low = int(np.floor((in_sp[-1] - out_sp[-1]) / 2))
    z_crop_high = in_sp[-1]-int(np.ceil((in_sp[-1] - out_sp[-1]) / 2))

    #Extract:
    if nd == 3:
        data_crop = data[x_crop_low:x_crop_high, y_crop_low:y_crop_high, z_crop_low:z_crop_high]
    elif nd == 4:
        data_crop = data[:, x_crop_low:x_crop_high, y_crop_low:y_crop_high, z_crop_low:z_crop_high]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop
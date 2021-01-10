import numpy as np
import torch
import sys
from scipy.stats import norm

def num2vect(x, bin_range, bin_step, sigma):
    """
    Function to convert a numerical vector x (hard label) into a label used by the model 
    (distribution over bins), e.g. the age 71.6 is converted into bin "10" (sigma=0) 
    or a distribution over age bins with mean 10 and variance sigma.
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
    bin_length = bin_range[1]-bin_range[0]
    if not bin_length % bin_step == 0:
        sys.exit("bin's range should be divisible by bin_step!")
        return -1

    bin_number = int(bin_length / bin_step)
    bin_centers = bin_range[0] + float(bin_step) / 2 + bin_step * torch.arange(bin_number)

    #If sigma is zero, set v to be index such that bin_centers[v] is closest
    #to x:
    if sigma == 0:
        abs_diff=torch.abs(x[:,None]-bin_centers[None,:])
        v=torch.argmin(abs_diff,dim=1)
        return v, bin_centers
    
    #If sigma is greater than zero, then return the probability of 
    #a bin under the normal distribution:
    elif sigma > 0:
        x1=bin_centers - float(bin_step) / 2 #Left bin boundary
        x2=bin_centers + float(bin_step) / 2 #Right bin boundary
        dist=torch.distributions.Normal(loc=0.,scale=sigma) 
        v=dist.cdf(x2[None,:]-x[:,None])-dist.cdf(x1[None,:]-x[:,None]) #Probability of bin-interval 
        return v, bin_centers
    else:
        sys.exit("Sigma must be either >=0.")

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
    x_crop_low = int(np.floor((in_sp[-1] - out_sp[-1]) / 2))
    x_crop_high = int(np.ceil((in_sp[-1] - out_sp[-1]) / 2))
    y_crop_low = int(np.floor((in_sp[-2] - out_sp[-2]) / 2))
    y_crop_high = int(np.ceil((in_sp[-2] - out_sp[-2]) / 2))
    z_crop_low = int(np.floor((in_sp[-3] - out_sp[-3]) / 2))
    z_crop_high = int(np.ceil((in_sp[-3] - out_sp[-3]) / 2))

    #Extract:
    if nd == 3:
        data_crop = data[z_crop_low:-z_crop_high, y_crop_low:-y_crop_high, x_crop_low:-x_crop_high]
    elif nd == 4:
        data_crop = data[:, z_crop_low:-z_crop_high, y_crop_low:-y_crop_high, x_crop_low:-x_crop_high]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop



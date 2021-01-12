import torch.nn as nn
import torch 

def my_KLDivLoss(x, y):
    """
    Input: 
        x  - torch.tensor - shape (n, m) - log-probabilities!
        y  - torch.tensor - shape (n, m) - probabilities
    Output:
        loss - float
    Returns KL-Divergence KL(y||x)
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to ensure
    numerical stability ("log(0)-problem")
    """
    loss_func = nn.KLDivLoss(reduction='batchmean')
    y += 1e-16
    loss = loss_func(x, y) 
    return loss

def MSE(x,y):
    '''
    x,y - torch.Tensor - arbitrary but similar shape
    '''
    MSE=torch.mean((x-y)**2)
    return(MSE)

def MAE(x,y):
    '''
    x,y - torch.Tensor - arbitrary but similar shape
    '''
    MAE=torch.mean(torch.abs(x-y))
    return(MAE)

def pred_from_dist(log_probs,bin_centers):
    '''
    Input:
        log_probs - torch.Tensor - shape (batch_size,length(bin_centers))/(bin_centers) - logarithmn of probability weights
        bin_centers - list of ints - centers of the (age) bins
    Output: means - torch.Tensor - shape (batch_size)/[1]
    '''
    probs=torch.exp(log_probs)
    bin_centers_=torch.tensor(bin_centers,dtype=probs.dtype)
    means=torch.matmul(probs,bin_centers_)
    return(means)

def eval_MAE(log_probs,bin_centers,target):
    preds=pred_from_dist(log_probs,bin_centers)
    return MAE(preds,target)
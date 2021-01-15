import torch.nn as nn
import torch 

def my_KLDivLoss(log_probs, target_probs,bin_centers=None):
    """
    Input: 
        log_probs  - torch.tensor - shape (n, m) - log-probabilities!
        y  - torch.tensor - shape (n, m) - probabilities
    Output:
        loss - float
    Returns KL-Divergence KL(y||log_probs)
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to ensure
    numerical stability ("log(0)-problem")
    """
    loss_func = nn.KLDivLoss(reduction='batchmean')
    target_probs += 1e-10
    log_probs=torch.log(1e-10+torch.exp(log_probs))
    loss = loss_func(log_probs, target_probs)  
    return loss

def my_MAELoss(log_probs, target_probs,bin_centers=None):
    """
    Input: 
        log_probs  - torch.tensor - shape (n, m) - log-probabilities!
        y  - torch.tensor - shape (n, m) - probabilities
    Output:
        loss - float
    Returns MSE between the predictions/
    """
    pred=pred_from_dist(log_probs,bin_centers)
    true_label=torch.matmul(target_probs,bin_centers)
    mae=MAE(pred,true_label)
    return(mae)

def MSE(x,y):
    '''
    x,y - torch.Tensor - arbitrary but similar shape
    '''
    mse=torch.mean((x-y)**2)
    return(mse)

def MAE(x,y):
    '''
    x,y - torch.Tensor - arbitrary but similar shape
    '''
    mae=torch.mean(torch.abs(x-y))
    return(mae)

def pred_from_dist(log_probs,bin_centers):
    '''
    Input:
        log_probs - torch.Tensor - shape (batch_size,length(bin_centers))/(bin_centers) - logarithmn of probability weights
        bin_centers - list of ints - centers of the (age) bins
    Output: means - torch.Tensor - shape (batch_size)/[1]
    '''
    probs=torch.exp(log_probs)
    #bin_centers_=torch.tensor(bin_centers,dtype=probs.dtype)
    means=torch.matmul(probs,bin_centers)#_)
    return(means)

def give_bin_eval(bin_centers):
    def eval_MAE(log_probs,target,bin_centers=bin_centers):
        preds=pred_from_dist(log_probs=log_probs,bin_centers=bin_centers)
        mae=MAE(preds,target)
        return(mae)
    return(eval_MAE)
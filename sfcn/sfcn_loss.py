import torch.nn as nn
import torch 
import sys

def MSE(x:torch.Tensor, y:torch.Tensor):
    '''
    x,y - arbitrary but same shape
    '''
    mse=torch.mean((x-y)**2)
    return(mse)

def MAE(x:torch.Tensor,y:torch.Tensor):
    '''
    x,y - arbitrary but same shape
    '''
    mae=torch.mean(torch.abs(x-y))
    return(mae)

def pred_from_dist(log_probs:torch.Tensor,bin_centers: torch.Tensor):
    '''
    Input:
        log_probs - shape (batch_size,n)  - logarithmn of probability weights
        bin_centers - shape (n) - center of bins 
    Output: means - torch.Tensor - shape (batch_size)/[1]
    '''
    probs=torch.exp(log_probs)
    means=torch.matmul(probs,bin_centers)
    return(means)


def give_my_loss_func(kwd):
    if kwd['type']=='kl':
        def my_kld(log_probs, target,bin_centers=kwd['bin_centers']):
            """
            Input: 
                log_probs  - torch.tensor - shape (n, m) - log-probabilities!
                target  - torch.tensor - shape (n, m) - probabilities
            Output:
                loss - float
            Returns KL-Divergence KL(target_probs||log_probs)
            Different from the default PyTorch nn.KLDivLoss in that
            a) the result is averaged by the 0th dimension (Batch size)
            b) the y distribution is added with a small value (1e-16) to ensure
            numerical stability ("log(0)-problem")
            """
            loss_func = nn.KLDivLoss(reduction='batchmean')
            target += 1e-10
            loss = loss_func(log_probs, target)  
            return loss
        return(my_kld)
        
    elif kwd['type']=='mae':
        
        def my_mae(log_probs, target,bin_centers=kwd['bin_centers'],probs=kwd['probs']):
            """
            Input: 
                log_probs  - torch.tensor - shape (n, m) - log-probabilities!
                target  - torch.tensor - if probs: shape (n, m) - probabilities
                                         else: shape (n) - targets
            Output:
                loss - float
            Returns MAE between the predictions/
            """
            bin_centers_=bin_centers.to(log_probs.device)
            pred=pred_from_dist(log_probs,bin_centers_)
            true_label=torch.matmul(target,bin_centers_) if probs else target
            mae=MAE(pred,true_label)
            return(mae)
        
        return(my_mae)

    elif kwd['type']=='ent':
        def my_ent(log_probs, target):
            """
            Input: 
                log_probs  - torch.tensor - shape (n, p) - probabilities of classes
                target  - torch.tensor - shape (n,p) - one-hot encoding of class
            Output:
                loss - float - cross entropy
            """
            cross_entr=-(log_probs*target).sum(dim=1).mean()
            return(cross_entr)
        return(my_ent)
    
    elif kwd['type']=='acc':
        def my_acc(log_probs,target,thresh=kwd['thresh']):
            '''
            log_probs - torch.Tensor - shape (batch_size,2)
            target - torch.Tensor - shape (batch_size)
            '''
            log_tresh=torch.log(torch.tensor(thresh))
            above_thresh=(log_probs[:,1]>log_tresh).float()
            correct_vec=above_thresh*target+(1-above_thresh)*(1-target)
            return(correct_vec.float().mean())
        return(my_acc)
    else: 
        sys.exit("Unknown loss function.")





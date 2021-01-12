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
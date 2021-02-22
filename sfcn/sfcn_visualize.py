
def compute_activation(model,x, filter_index,device=None):
    '''
        model - SFCN 
        x - input MRI 
        filter_index - int 
    '''
    if device is not None: 
        x=x.to(device)
    activation = model.module.feature_extractor(x)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return torch.mean(filter_activation)

def maximize_activation(model,x,filter_index):
    x.requires_grad=True


x=torch.randn(4)
maximize_activation(None,x,None)
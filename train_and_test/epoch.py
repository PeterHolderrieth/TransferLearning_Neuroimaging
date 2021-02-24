#DOES MODEL OPTIMIZATION REALLY HAPPEN INPLACE?
import torch 
#Function to go one epoch and evaluate results:
def go_one_epoch(state, model, loss_func, device, data_loader, optimizer, label_translater, eval_func=None):
    '''
    Function training the model over one epoch or evaluating the model over one epoch.
    Inputs: 
        state - string - specifies training or test state
        model - torch.nn.Module 
        loss_func - the loss function (average over batch)
        device - torch.device 
        data_loader - giving inputs and labels of model
        eval_func - function evaluating model performance by comparing model output with true label (average over batch)
        optimizer - optimizer torch.optim.NAME 
        label_translater - function transforming a "hard" label y into a "soft" label similar to the output of the model
    Output: 
        results - dictionary - giving evaluation metric (e.g. accuracy or MAE) and average loss 
    '''
    #Send model to device:
    #model=model.to(device)
    #DEBUG:
    #par_llayer=model.module.state_dict()['classifier.conv_6.weight'].flatten().cpu()

    #Set train or evaluation state:
    if state == 'train':
        model.train()
    elif state == 'test':
        model.eval()
    
    else:
        raise (f'train or test? Received {state}')

    # Initialize logging:
    n_total = 0
    loss_total = 0
    eval_total = 0
    
    #Go over data:
    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        n_batch = data.shape[0]
        label=label.squeeze().to(device)
        #Translate label into the same space as the outputs:
        target_probs= label_translater(label)
        
        if state == 'train':
            #Compute loss and gradient step:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(log_probs=output, target=target_probs)
            loss.backward()
            optimizer.step()
        else:
            #Compute loss without gradient step:
            with torch.no_grad():
                output = model(data)
                loss = loss_func(log_probs=output, target=target_probs)
        
        # Step Logging:
        n_total += n_batch
        loss_total += loss.detach().cpu().item() * n_batch 

        #Note that if state=='train', this is not necessarily the correct
        #error on the train set since we use dropout and batch normalization.
        if eval_func is not None:
            with torch.no_grad():
                #print("True labels: ", label)
                #print("Predictions:", torch.matmul(torch.exp(output),bin_centers))
                eval_= eval_func(log_probs=output, target=label)
                eval_total=eval_total+eval_.item()*n_batch        
        
    #DEBUG: Observe gradient descent in the parameters:
    #par_nlayer=model.state_dict()['classifier.conv_6.weight'].flatten().cpu()
    #abs_diff=torch.abs(par_llayer-par_nlayer)
    #print("Inner-Loop Maximum difference: ", abs_diff.max().item())
    #print("Inner-Loop Minimum difference: ", abs_diff.min().item())    
    #print("Inner-Loop Mean difference: ", abs_diff.mean().item())
    
    # Output Logging
    results = { 'eval': eval_total / n_total,
                'loss': loss_total / n_total
                }   
    return results
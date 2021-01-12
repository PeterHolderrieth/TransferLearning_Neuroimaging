#DOES MODEL OPTIMIZATION REALLY HAPPEN INPLACE?

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

    #Set train or evaluation state:
    if state == 'train':
        model.train()
    elif state == 'test':
        model.eval()
    else:
        raise (f'train or test? Received {state}')

    #Send model to device:
    model=model.to(device)

    # Initialize logging:
    output_all = list()
    label_all = list()
    n_total = 0
    loss_total = 0
    eval_total = 0
    
    #Go over data:
    for batch_idx, (data, label) in enumerate(data_loader):
        print("Batch: ", batch_idx)
        data = data.to(device)
        n_batch = data.shape[0]

        #Translate label into the same space as the outputs:
        target,bin_centers = label_translater(label)
        target=target.to(device)
        
        if state == 'train':
            #Compute loss and gradient step:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        else:
            #Compute loss without gradient step:
            with torch.no_grad():
                output = model(data)
                loss = loss_func(output, target)

        # Step Logging:
        n_total += n_batch
        loss_total += loss.detach().cpu().item() * n_batch 

        if eval_func is not None:
            eval_total = eval_func(output, label,bin_centers=bin_centers)*n_batch        
    
    # Output Logging
    results = { 'eval': eval_total / n_total,
                'loss': loss_total / n_total
                }   
    
    return results
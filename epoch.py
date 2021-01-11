
#Function to go one epoch and evaluate results:
def go_one_epoch(state, model, loss_func, unpack_func, device, data_loader, metric_func=None, optimizer=None,
                 record_output=False):
    '''
    state - string - specifies training or test state
    model - torch.nn.Module 
    loss_func - the loss function
    unpack_func - 
    device - torch.device 
    data_loader - torch.
    metric_func - 
    optimizer - 
    record_output - Boolean - 
    '''
    #Set train or evaluation state:
    if state == 'train':
        model.train()
    elif state == 'test':
        model.eval()
    else:
        raise (f'train or test? Received {state}')

    # Initialize logging:
    output_all = list()
    label_all = list()
    n_total = 0
    loss_total = 0
    metric_total = 0
    
    #Go over data:
    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        n_batch = data.shape[0]

        if state == 'train':
            target = unpack_func(label, device)
            optimizer.zero_grad()
            output = model(data)
            loss, loss_list = loss_func(output, target, device)
            loss.backward()
            optimizer.step()
        elif state == 'test':
            with torch.no_grad():
                target = unpack_func(label, device)
                output = model(data)
                loss, loss_list = loss_func(output, target, device)

        # Step Logging:
        n_total += n_batch
        ???WHY TIMES n_batch?
        loss_total += loss.detach().cpu().item() * n_batch 

        if metric_func is not None:
            metric = metric_func(output, label)
            metric = metric * n_batch
            metric_total += metric
        if record_output is True:
            output_all.append(list(out_.detach().cpu() for out_ in output))
            label_all.append(list(l_.detach().cpu() for l_ in label))
    
    
    # Output Logging
    results = { 'metric': metric_total / n_total,
                'loss': loss_total / n_total
                }    
    if record_output is True:
        results['output'] = output_all
        results['label'] = label_all
    
    return results
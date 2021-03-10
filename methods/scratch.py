import sys
sys.path.append('../')

from sfcn.sfcn_load import give_fresh_sfcn
from sfcn.sfcn_train import sfcn_train
from sfcn.sfcn_test import sfcn_test 

def load_from_scratch_sfcn(bin_min,bin_max,dropout,channel_number): 
    return give_fresh_sfcn(bin_min,bin_max,dropout,channel_number)

def train_from_scratch_sfcn(train_loader,val_loader,hps): 
    
    model=load_from_scratch_sfcn(hps['bin_min'],hps['bin_max'],hps['dropout'],hps['channel_number'])
    model.module.train_full_model()

    info_start="Full model is being trained with random initialization."
    info_end="Full model was being trained with random initialization."

    meter=sfcn_train(model,   train_loader=train_loader, 
                        val_loader=val_loader,
                        bin_min=hps['bin_min'], 
                        bin_max=hps['bin_max'],
                        space=hps['space'],
                        loss_met=hps['loss_met'],
                        eval_met=hps['eval_met'],
                        bin_step=hps['bin_step'],
                        sigma=hps['sigma'],
                        n_epochs=hps['n_epochs'],
                        optim_type=hps['optim_type'],
                        lr=hps['lr'],
                        weight_decay=hps['weight_decay'],
                        momentum=hps['momentum'],
                        scheduler_type=hps['scheduler_type'],
                        epoch_dec=hps['epoch_dec'],
                        gamma_dec=hps['gamma_dec'],
                        threshold=hps['threshold'],
                        print_corr=hps['print_corr'],
                        info_start=info_start,
                        info_end=info_end)
    return(model)

def test_from_scratch_sfcn(test_loader,hps,file_path=None,model=None):
    if file_path is not None:
        model=load_from_scratch_sfcn(hps['bin_min'],hps['bin_max'],hps['dropout'],hps['channel_number'])
        model.load_state_dict(file_path)
    elif model is not None:
        pass 
    else: 
        sys.exit("Neither model nor file path is given.")

    model.train_nothing()
    model.eval()
    
    info_start="Model loaded from %s is being tested."%file_path
    info_end="Model loaded from %s has been tested."%file_path

    return sfcn_test(model=model,
                test_loader=test_loader,
                bin_min=hps['bin_min'],
                bin_max=hps['bin_max'],
                space=hps['space'],
                loss_met=hps['loss_met'],
                eval_met=hps['eval_met'],
                bin_step=hps['bin_step'],
                sigma=hps['sigma'],
                n_epochs=3,
                print_corr=True,
                info_start=info_start,
                info_end=info_end)

'''
Utilities for later us:
writer.add_graph(model, input_data)
writer.close()
model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
print(torch.cuda.device_count()) 
'''
#AGE_RANGE=[40,96] #Age range of data
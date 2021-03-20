import sys
import torch
sys.path.append('../')

from sfcn.sfcn_load import give_pretrained_sfcn
from sfcn.sfcn_train import sfcn_train
from sfcn.sfcn_test import sfcn_test

def load_full_sfcn_preloaded(run,task,bin_min,bin_max,reinit_with_scaling=None):
    
    model=give_pretrained_sfcn(run,task)
    model.module.change_output_dim(bin_max-bin_min)
    if reinit_with_scaling is not None:
        model.module.reinit_final_layers_pres_scale(reinit_with_scaling)
    return model 


def train_full_sfcn_preloaded(train_loader,val_loader,hps): 
    model=load_full_sfcn_preloaded(hps['run'],hps['task'],hps['bin_min'],hps['bin_max'],hps.get('reinit_with_scaling',None))
    model.module.train_full_model()

    info_start="Full model is being trained with weights loaded from pre-trained model."
    info_end="Full model was being trained with weights loaded from pre-trained model."

    sfcn_train(model,   train_loader=train_loader, 
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
    
    return model

def test_full_sfcn_preloaded(test_loader,hps,file_path=None,model=None):
        
    if file_path is not None:
        model=load_full_sfcn_preloaded(hps['run'],hps['task'],hps['bin_min'],hps['bin_max'])
        model.load_state_dict(torch.load(file_path))
    elif model is not None:
        pass 
    else: 
        sys.exit("Neither model nor file path is given.")

    model.module.train_nothing()
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
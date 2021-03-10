import sys
sys.path.append('../')

from sfcn.sfcn_load import give_pretrained_sfcn
from sfcn.sfcn_train import sfcn_train
from sfcn.sfcn_test import sfcn_test

def load_step_sfcn_preloaded(run,task,bin_min,bin_max):
    
    model=give_pretrained_sfcn(run,task)
    model.module.change_output_dim(bin_max-bin_min)

    return model 

def train_step_sfcn_preloaded(train_loader,val_loader,hps): 

    model=load_step_sfcn_preloaded(hps['run'],hps['task'],hps['bin_min'],hps['bin_max'])
    model.module.train_only_final_layers(hps['n_layer_ft'])

    info_start="Final layer of model is being trained."
    info_end="Final layer of model was being trained."

    sfcn_train(model,   train_loader=train_loader, 
                        val_loader=val_loader,
                        bin_min=hps['bin_min'], 
                        bin_max=hps['bin_max'],
                        space=hps['space'],
                        loss_met=hps['loss_met'],
                        eval_met=hps['eval_met'],
                        bin_step=hps['bin_step'],
                        sigma=hps['sigma'],
                        n_epochs=hps['n_epochs_ll'],
                        optim_type=hps['optim_type_ll'],
                        lr=hps['lr_ll'],
                        weight_decay=hps['weight_decay_ll'],
                        momentum=hps['momentum_ll'],
                        scheduler_type=hps['scheduler_type_ll'],
                        epoch_dec=hps['epoch_dec_ll'],
                        gamma_dec=hps['gamma_dec_ll'],
                        threshold=hps['threshold_ll'],
                        print_corr=hps['print_corr'],
                        info_start=info_start,
                        info_end=info_end)
    
    model.module.train_full_model()
    
    info_start="Full model is being trained."
    info_end="Full model was being trained."

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

def test_step_sfcn_preloaded(test_loader,hps,file_path=None,model=None):
    if file_path is not None:
        model=load_step_sfcn_preloaded(hps['run'],hps['task'],hps['bin_min'],hps['bin_max'])
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
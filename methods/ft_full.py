import sys
sys.path.append('../')

from sfcn.sfcn_load import give_pretrained_sfcn
from sfcn.sfcn_train import sfcn_train

def train_sfcn_preloaded(train_loader,val_loader,hps): 
    
    model=give_pretrained_sfcn(hps['run'],hps['task'])
    model.module.change_output_dim(hps['bin_max']-hps['bin_min'])
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
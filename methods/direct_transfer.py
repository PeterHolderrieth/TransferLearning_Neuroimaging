import sys
sys.path.append('../')

from sfcn.sfcn_load import give_pretrained_sfcn
from sfcn.sfcn_test import sfcn_test

def test_sfcn_preloaded(test_loader,hps): 
    
    model=give_pretrained_sfcn(hps['run'],hps['task'])
    model.module.change_output_dim(hps['bin_max']-hps['bin_min'])
    model.module.train_full_model()

    info_start="Full model is being tested with weights loaded from pre-trained model."
    info_end="Full model was being tested with weights loaded from pre-trained model."

    return sfcn_test(model,   train_loader=train_loader, 
                        val_loader=val_loader,
                        bin_min=hps['bin_min'], 
                        bin_max=hps['bin_max'],
                        space=hps['space'],
                        loss_met=hps['loss_met'],
                        eval_met=hps['eval_met'],
                        bin_step=hps['bin_step'],
                        sigma=hps['sigma'],
                        n_epochs=hps['n_epochs'],
                        print_corr=hps['print_corr'],
                        info_start=info_start,
                        info_end=info_end)
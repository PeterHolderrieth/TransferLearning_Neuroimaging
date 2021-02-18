from sfcn.sfcn_utils import give_metrics
from train_and_test.testing import run_test

def sfcn_test(model,   test_loader, 
                        bin_min:int, 
                        bin_max:int,
                        space: str, 
                        loss_met: str, 
                        eval_met:str, 
                        bin_step: int,
                        sigma: float,
                        n_epochs: int,
                        print_corr: bool,
                        info_start: str,
                        info_end: str
                        ):
    
    loss_func,eval_func,label_translater=give_metrics(bin_min=bin_min, 
                                                        bin_max=bin_max,
                                                        space=space, 
                                                        loss_met=loss_met, 
                                                        eval_met=eval_met, 
                                                        bin_step=bin_step,
                                                        sigma=sigma)

    meter=run_test(model=model,
                        test_loader=test_loader,
                        loss_func=loss_func,
                        eval_func=eval_func,
                        label_translater=label_translater,
                        n_epochs=n_epochs,
                        print_corr=print_corr,
                        info_start=info_start,
                        info_end=info_end)
    return meter
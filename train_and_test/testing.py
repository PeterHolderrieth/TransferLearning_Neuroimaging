import datetime
import torch
import numpy as np 
from train_and_test.epoch import go_one_epoch
from train_and_test.train_utils import TestMeter as TM
from train_and_test.train_utils import print_sep_line

def test(model,test_loader,loss_func,eval_func,label_translater,n_epochs,device):
    meter=TM()
    for epoch in range(n_epochs):
        #Go one epoch:
        results_test=go_one_epoch('test',model=model,
                                    loss_func=loss_func,
                                    device=device,
                                    data_loader=test_loader,
                                    optimizer=None,
                                    label_translater=label_translater,
                                    eval_func=eval_func)

        #Update logging:
        meter.update(test_loss_it=results_test['loss'],
                     test_eval_it=results_test['eval'])
        
        loss_name=loss_func.__name__[-3:]
        ev_name=eval_func.__name__[-3:]

        loss_std=torch.tensor(meter.test_loss.vec).std().item()
        ev_std=torch.tensor(meter.test_eval.vec).std().item()

    print(("test %s mean: %.5f | test %s std: %.5f |"+
            "test %s mean:  %.5f |test %s std:  %.5f |")%(loss_name,meter.test_loss.avg
                ,loss_name,loss_std,ev_name,meter.test_eval.avg,ev_name,ev_std))

    return(meter)

def run_test(model,test_loader,loss_func,eval_func,label_translater,n_epochs,info_start=None,info_end=None,print_corr=True):

    #Load the device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs: ", torch.cuda.device_count())
    model=model.to(device)

    print()
    print(info_start)
    print("Start: ", datetime.datetime.today())
    print_sep_line()

    #Train the model:
    meter=test(model=model,
                test_loader=test_loader,
                loss_func=loss_func,
                eval_func=eval_func,
                label_translater=label_translater,
                device=device,
                n_epochs=n_epochs)
                
    print_sep_line() 
    print("Finished: ", datetime.datetime.today())
    print()

    if print_corr:
        loss_arr=np.array(meter.test_loss.vec)
        eval_arr=np.array(meter.test_eval.vec)
        corr=np.corrcoef(loss_arr,eval_arr)[0,1]
        
        print("Correlation between train loss and evaluation:", "%.3f"%corr)
        print() 

    print_sep_line()
    
    return(meter)





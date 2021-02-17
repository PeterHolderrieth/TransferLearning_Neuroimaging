#import utils.TrainMeter as TM
import datetime
from epoch import go_one_epoch
from utils import TrainMeter as TM
from utils import give_optimizer,give_lr_scheduler
from utils import print_sep_line

def train(model,train_loader,val_loader,loss_func,eval_func,label_translater,device,n_epochs,optimizer,scheduler,print_every=1,len_rvg=5):
    meter=TM(len_rvg=len_rvg)
    for epoch in range(n_epochs):
        
        #Get learning rate:    
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        
        #Go one epoch:
        results_tr=go_one_epoch('train',model=model,
                                    loss_func=loss_func,
                                    device=device,
                                    data_loader=train_loader,
                                    optimizer=optimizer,
                                    label_translater=label_translater,
                                    eval_func=eval_func)
        #Go one epoch:
        results_val=go_one_epoch('test',model=model,
                                    loss_func=loss_func,
                                    device=device,
                                    data_loader=val_loader,
                                    optimizer=optimizer,
                                    label_translater=label_translater,
                                    eval_func=eval_func)

        #Update logging:
        meter.update(tr_loss_it=results_tr['loss'],
                    tr_eval_it=results_tr['eval'],
                    val_loss_it=results_val['loss'],
                    val_eval_it=results_val['eval'])
        
        loss_name=loss_func.__name__[-3:]
        ev_name=eval_func.__name__[-3:]
        #Print update:
        if epoch%print_every==0:
            print(("|epoch: %3d | lr: %.3f |"+ 
                    "train %s: %.5f |train %s ravg: %.5f |"+
                    "train %s:  %.5f |train %s ravg:  %.5f |"+
                    "val %s: %.5f |val %sravg: %.5f |"+
                    "val %s:  %.5f |val %s ravg:  %.5f |")%(epoch,
                    lr,loss_name,meter.tr_loss.val,loss_name,meter.tr_loss.run_avg,ev_name,meter.tr_eval.val,ev_name,meter.tr_eval.run_avg,
                    loss_name,meter.val_loss.val,loss_name,meter.val_loss.run_avg,ev_name,meter.val_eval.val,ev_name,meter.val_eval.run_avg))

        scheduler.step(meter.tr_loss.val)

    return(meter)

def run_training(model, train_loader,
                        val_loader,
                        loss_func:callable, 
                        eval_func:callable,
                        label_translater:callable,
                        n_epochs: int,
                        optim_type:str,
                        lr:float,
                        weight_decay: float,
                        momentum:float,
                        scheduler_type:str,
                        epoch_dec:int,
                        gamma_dec:float,
                        threshold:float=None,
                        print_every:int =1,
                        print_corr: bool =True,
                        info_start: str =None,
                        info_end: str =None)

    #Load the device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs: ", torch.cuda.device_count())
    model=model.to(device)

    #Get parameter optimizer and learning rate scheduler:
    optimizer=give_optimizer(optim_type,model,lr,weight_decay,momentum)
    scheduler=give_lr_scheduler(scheduler_type,optimizer,epoch_dec,gamma_dec,threshold)
    
    
    print()
    print(info_start)
    print("Start: ", datetime.datetime.today())
    print_sep_line()

    #Train the model:
    meter=train(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_func=loss_func,
                eval_func=eval_func,
                label_translater=label_translater,
                device=device,
                n_epochs=n_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                print_every=print_every,
                len_rvg=len_rvg)
                
    print_sep_line() 
    print("Finished: ", datetime.datetime.today())
    print()

    if print_corr:
        loss_arr=np.array(meter.tr_loss.vec)
        eval_arr=np.array(meter.tr_eval.vec)
        corr=np.corrcoef(loss_arr,eval_arr)[0,1]
        
        print("Correlation between train loss and evaluation:", "%.3f"%corr)
        print() 

    print_sep_line()
    
    return(meter)





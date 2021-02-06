#import utils.TrainMeter as TM
from epoch import go_one_epoch
from utils import TrainMeter as TM

def train(model,n_epochs,loss_func,device,train_loader,val_loader,optimizer,scheduler,label_translater,eval_func,print_every=1,len_rvg=3):
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

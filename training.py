import utils.TrainMeter as TM
from epoch import go_one_epoch

def train(model,n_epochs,loss_func,device,train_loader,val_loader,optimizer,label_translater,eval_func,print_every=1,len_avg=10):
    meter=TM(len_avg=len_avg)
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
        
        #Print update:
        if epoch%print_every==0:
            print(("|epoch: %3d | lr: %.3f |"+ 
                    "train loss: %.5f |train loss ravg: %.5f |"+
                    "train MAE:  %.5f |train MAE ravg:  %.5f |"+
                    "val loss: %.5f |val loss ravg: %.5f |"+
                    "val MAE:  %.5f |val MAE ravg:  %.5f |")%(epoch,
                    lr,meter.tr_loss.val,meter.tr_loss.run_avg,meter.tr_eval.val,meter.tr_eval.run_avg,
                    meter.val_loss.val,meter.val_loss.run_avg,meter.val_eval.val,meter.val_eval.run_avg))

        scheduler.step(meter.tr_loss.val)

        return(meter)

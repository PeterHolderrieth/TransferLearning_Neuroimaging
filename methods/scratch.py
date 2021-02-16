from sfcn.sfcn_load import give_fresh_sfcn
from sfcn.sfcn_train import sfcn_train

def train_sfcn_from_scratch(hps): 
    
    model=give_fresh_sfcn(hps['bin_min'],hps['bin_max'],hps['dropout'])
    model.module.train_full_model()

    info_start="Full model is being trained with random initialization."
    info_end="Full model was being trained with random initialization."

    sfcn_train(model,   bin_min:int, 
                        bin_max:int,
                        space: str, 
                        loss_met: str, 
                        eval_met:str, 
                        bin_step: int,
                        sigma: float,
                        n_epochs : int ,
                        optim_type: str,
                        lr: float,
                        weight_decay: float,
                        momentum: float,
                        scheduler_type: str,
                        epoch_dec: int,
                        gamma_dec: float,
                        threshold: float,
                        print_corr: bool,
                        info_start=info_start,
                        info_end=info_end):
    


'''
Utilities for later us:
writer.add_graph(model, input_data)
writer.close()
model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
print(torch.cuda.device_count()) 
'''
#AGE_RANGE=[40,96] #Age range of data
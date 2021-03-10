#Parsing libraries:
import argparse
import json 
import sys 
import os 
import torch 

from methods.elastic import elastic_experiment, elastic_grid_search
from methods.scratch import train_from_scratch_sfcn
from methods.ft_full import train_sfcn_preloaded
from methods.ft_final import train_final_sfcn_preloaded
from methods.ft_step import train_step_sfcn_preloaded
from methods.direct_transfer import test_sfcn_preloaded

from data.oasis.load_oasis3 import give_oasis_data
from data.abide.load_abide import give_abide_data
from data.ixi.load_ixi import give_ixi_data

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(TEST=False)

ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="'debug' or 'full'.")
ap.add_argument("-con", "--CONFIG", type=str, required=True,help="Path to 'config' file.")
ap.add_argument("-test", "--TEST", type=bool, required=False,help="Whether file should be executed or only tested.")

#Get arguments:
ARGS = vars(ap.parse_args())

#Read config file:
with open(ARGS["CONFIG"], "r") as read_file:
    config = json.load(read_file)

#Extract task, data, and method:
task=config['experiment']['task']
data=config['experiment']['data']
balance=config['experiment']['balance']
method=config['experiment']['method']
preprocessing=config['experiment']['preprocessing']
share=config['experiment']['share']


record_config=config['record']

#Extract hyperparameters for experiment:
config_setup=config[method][task][data]
hps=config_setup['hps']
computing=config_setup['computing']

#Set debug flag:
if ARGS['DEBUG']=='debug':
    debug=True
    hps['batch']=2
    if 'n_epochs' in list(hps.keys()):
        hps['n_epochs']=3    
    if 'n_epochs_ll' in list(hps.keys()):
        hps['n_epochs_ll']=3

elif ARGS['DEBUG']=='full':
    debug=False 
else: 
    sys.exit("Unvalid debug flag.")

if not ARGS['TEST']:
    #Get the data files: 
    if data=='oasis':
        _,train_loader=give_oasis_data('train', batch_size=hps['batch'],
                                                num_workers=computing['n_workers'],
                                                shuffle=True,
                                                debug=debug,
                                                preprocessing=preprocessing,
                                                task=task,
                                                share=share)

        _,val_loader=give_oasis_data('val', batch_size=hps['batch'],
                                            num_workers=computing['n_workers'],
                                            shuffle=True,
                                            debug=debug,
                                            preprocessing='min',
                                            task=task)

    elif data=='abide':
        _,train_loader=give_abide_data('train', batch_size=hps['batch'],
                                                num_workers=computing['n_workers'],
                                                shuffle=True,
                                                debug=debug,
                                                preprocessing=preprocessing,
                                                task=task,
                                                share=share,
                                                balance=balance)

        _,val_loader=give_abide_data('val', batch_size=hps['batch'],
                                            num_workers=computing['n_workers'],
                                            shuffle=True,
                                            debug=debug,
                                            preprocessing='min',
                                            task=task)
    elif data=='ixi':
        _,train_loader=give_ixi_data('train', batch_size=hps['batch'],
                                                num_workers=computing['n_workers'],
                                                shuffle=True,
                                                debug=debug,
                                                preprocessing=preprocessing,
                                                task=task,
                                                share=share,
                                                balance=balance)

        _,val_loader=give_ixi_data('val', batch_size=hps['batch'],
                                            num_workers=computing['n_workers'],
                                            shuffle=True,
                                            debug=debug,
                                            preprocessing='min',
                                            task=task)
    else: 
        sys.exit("Unknown data files.")   

    if method=='elastic':
        elastic_experiment(train_loader,val_loader,hps)

    elif method=='elastic_grid':
        elastic_grid_search(train_loader,val_loader,hps)

    elif method=='scratch':
        model=train_from_scratch_sfcn(train_loader,val_loader,hps)

    elif method=='ft_full':
        model=train_sfcn_preloaded(train_loader,val_loader,hps)

    elif method=='ft_final':
        model=train_final_sfcn_preloaded(train_loader,val_loader,hps)

    elif method=='ft_step':
        model=train_step_sfcn_preloaded(train_loader,val_loader,hps)

    elif method=='direct_transfer':
        test_sfcn_preloaded(val_loader,hps)
    else: 
        sys.exit("Unknown method.")

    #Check whether model should be saved:
    model_save=record_config.get('model_save',False)
    if model_save and method!='elastic' and method!='elastic_grid' and method!='direct_transfer':
        file_path=os.path.join(record_config['model_save_folder'],record_config['model_save_name']+'.p')
        torch.save(model.state_dict(),file_path)


    test_after_training=config['experiment'].get('test_after_training',False)

    if test_after_training:
        #Get the data files: 
        if data=='oasis':
            _,test_loader=give_oasis_data('test0', batch_size=hps['batch'],
                                                    num_workers=computing['n_workers'],
                                                    shuffle=True,
                                                    debug=debug,
                                                    preprocessing='min',
                                                    task=task,
                                                    share=share)

        elif data=='abide':
            _,test_loader=give_abide_data('test', batch_size=hps['batch'],
                                                    num_workers=computing['n_workers'],
                                                    shuffle=True,
                                                    debug=debug,
                                                    preprocessing='min',
                                                    task=task,
                                                    share=share,
                                                    balance=balance)

        elif data=='ixi':
            _,test_loader=give_ixi_data('test', batch_size=hps['batch'],
                                                    num_workers=computing['n_workers'],
                                                    shuffle=True,
                                                    debug=debug,
                                                    preprocessing='min',
                                                    task=task,
                                                    share=share,
                                                    balance=balance)

        if train and method!='elastic' and method!='elastic_grid':
            pass
else: 
    pass
#Parsing libraries:
import argparse
import json 
import sys 
import os 
import torch 
import pickle as pk

from methods.elastic import elastic_experiment, elastic_grid_search, test_elastic
from methods.scratch import train_from_scratch_sfcn, test_from_scratch_sfcn
from methods.ft_full import train_full_sfcn_preloaded, test_full_sfcn_preloaded
from methods.ft_final import train_final_sfcn_preloaded, test_final_sfcn_preloaded
from methods.ft_step import train_step_sfcn_preloaded, test_step_sfcn_preloaded
from methods.direct_transfer import test_sfcn_preloaded
from sfcn.sfcn_test import sfcn_test

from data.load_dataset import give_dataset

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(TEST=False)

ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="'debug' or 'full'.")
ap.add_argument("-con", "--CONFIG", type=str, required=True,help="Path to 'config' file.")
ap.add_argument("-test", "--TEST", type=bool, required=False,help="If false (default), the model is trained/fitted. \
                                                                    If true, the model is tested (training has to be completed).")

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


#------------------------------------
#************Set debug flag*************
#-------------------------------------
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


if record_config.get('model_save',False) or ARGS['TEST']:
    #How to save filepath:    
    if method=='elastic' or method=='elastic_grid':
        ending='.pkl'
    else: 
        ending='.p'

    if debug: 
        ending='_debug'+ending

    directory=os.path.join(record_config['model_save_folder'],data,task)
    file_path=os.path.join(directory,record_config['model_save_name']+ending)

    if not os.path.exists(directory):
        os.makedirs(directory)

#-------------------------------------
#*************TRAINING*************
#-------------------------------------

if not ARGS['TEST']:
    #-------------------------------------
    #***********Load train data***********
    #-------------------------------------
    _,train_loader=give_dataset(data,'train', batch_size=hps['batch'],
                                            num_workers=computing['n_workers'],
                                            shuffle=True,
                                            debug=debug,
                                            preprocessing=preprocessing,
                                            task=task,
                                            share=share,
                                            balance=balance)

    _,val_loader=give_dataset(data,'val', batch_size=hps['batch'],
                                        num_workers=computing['n_workers'],
                                        shuffle=True,
                                        debug=debug,
                                        preprocessing='min',
                                        task=task,
                                        share=1.0)
    #-------------------------------------
    #*************Run method*************
    #-------------------------------------

    if method=='elastic':
        _,model_dict=elastic_experiment(train_loader,val_loader,hps)
        
    elif method=='elastic_grid':
        elastic_grid_search(train_loader,val_loader,hps)

    elif method=='scratch':
        model=train_from_scratch_sfcn(train_loader,val_loader,hps)

    elif method=='ft_full':
        model=train_full_sfcn_preloaded(train_loader,val_loader,hps)

    elif method=='ft_final':
        model=train_final_sfcn_preloaded(train_loader,val_loader,hps)

    elif method=='ft_step':
        model=train_step_sfcn_preloaded(train_loader,val_loader,hps)

    elif method=='direct_transfer':
        test_sfcn_preloaded(val_loader,hps)
    else: 
        sys.exit("Unknown method.")


    #save model and set training completed:
    if record_config.get('model_save',False):
        if method in ['scratch','ft_full','ft_step','ft_final']:
            torch.save(model.state_dict(),file_path)
            config['record']['model_has_been_saved']=True

        elif method=='elastic':
            pk.dump(model_dict, open(file_path,"wb"))
            config['record']['model_has_been_saved']=True
            #sys.exit("So far, it is not possible to save the elastic net model.")

        elif method=='direct_transfer':
            sys.exit("It is not necessary to save the 'direct transfer' model. It has not been trained.")
        
        else:
            sys.exit("Unknown method.")

    #Set training completed and save (if not in debug mode)
    config['experiment']['training_completed']=True
    if not debug:
        with open(ARGS["CONFIG"], "w") as configfile:
            print("Save configfile.")
            json.dump(config,configfile,indent=2)
    
    still_have_model=True
else: 
    still_have_model=False
#-------------------------------------
#*************TESTING*****************
    model_save=record_config.get('model_save',False)
#-------------------------------------
if ARGS['TEST'] or config['experiment'].get('test_after_training',False):
    if not config['experiment'].get('training_completed',False) and not debug:
        sys.exit("Training of the model was not completed. We should not test.")

    _,test_loader=give_dataset(data,'test', batch_size=hps['batch'],
                                        num_workers=computing['n_workers'],
                                        shuffle=True,
                                        debug=debug,
                                        preprocessing='min',
                                        task=task,
                                        share=1.0)
    if method=='elastic':
        if still_have_model:
            test_elastic(test_loader,hps['reg_method'],**model_dict)
        else: 
            model_dict=pk.load(open(file_path,"rb"))
            test_elastic(test_loader,hps['reg_method'],**model_dict)
            
    elif method=='elastic_grid':
        sys.exit("Testing is not enabled for elastic net grid search.")

    elif method=='scratch':
        if still_have_model:
            test_from_scratch_sfcn(test_loader,hps,model=model)
        else: 
            test_from_scratch_sfcn(test_loader,hps,file_path=file_path)

    elif method=='ft_full':
        if still_have_model:
            test_full_sfcn_preloaded(test_loader,hps,model=model)
        else: 
            test_full_sfcn_preloaded(test_loader,hps,file_path=file_path)

    elif method=='ft_final':
        if still_have_model:
            test_final_sfcn_preloaded(test_loader,hps,model=model)
        else: 
            test_final_sfcn_preloaded(test_loader,hps,file_path=file_path)

    elif method=='ft_step':
        if still_have_model:
            test_step_sfcn_preloaded(test_loader,hps,model=model)
        else: 
            test_step_sfcn_preloaded(test_loader,hps,file_path=file_path)

    elif method=='direct_transfer':
        test_sfcn_preloaded(test_loader,hps)
    else: 
        sys.exit("Unknown method.")

#Parsing libraries:
import argparse
import json 

from methods.elastic import elastic_experiment
#import methods.ft_final
#import methods.ft_full
#import methods.ft_full
#import methods.ft_full

from data.oasis.load_oasis3 import give_oasis_data

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults()

ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="'debug' or 'full'.")
ap.add_argument("-con", "--CONFIG", type=str, required=True,help="Path to 'config' file.")

#Get arguments:
ARGS = vars(ap.parse_args())

#Read config file:
with open(ARGS["CONFIG"], "r") as read_file:
    config = json.load(read_file)

#Extract task, data, and method:
task=config['experiment']['task']
data=config['experiment']['data']
method=config['experiment']['method']
preprocessing=config['experiment']['preprocessing']

#Extract hyperparameters for experiment:
config_setup=config[method][task][data]
hps=config_setup['hps']
computing=config_setup['computing']

#Set debug flag:
if ARGS['DEBUG']=='debug':
    debug=True
elif ARGS['DEBUG']=='full':
    debug=False 
else: 
    sys.exit("Unvalid debug flag.")


#Get the data files: 
if data=='oasis':
    _,train_loader=give_oasis_data('train', batch_size=hps['batch'],
                                            num_workers=computing['n_workers'],
                                            shuffle=True,
                                            debug=debug,
                                            preprocessing=preprocessing,
                                            task=task)

    _,val_loader=give_oasis_data('val', batch_size=hps['batch'],
                                        num_workers=computing['n_workers'],
                                        shuffle=True,
                                        debug=debug,
                                        preprocessing=preprocessing,
                                        task=task)
else: 
    sys.exit("Unknown data files.")   

if method=='elastic':
    space='binary' if task=='sex' else 'continuous'
    elastic_experiment(train_loader,hps,space)
else: 
    sys.exit("Unknown method.")
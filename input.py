import configparser
import sys 
import pandas as pd
import numpy as np
import os.path as osp
import os
from datetime import datetime



'''
Note: I would like to two codes:
1. An interactive session where based on suggestions, I interactively
give the inputs. So I am asked! This program asked about hyperparameters 
based on the task I submit and controls whether all values are correct. It
then writes this in config files. It also asks whether my program should 
be immediately executed (usually it should not.) This program should make
clear that every file I write will have a unique name (just add datetime)
and a unique .log file.
This program should also immediately write a bash script to submit the job.
Basically, after parsing the argument, there should be two files: 
the bash script to submit the job and the corresponding file (in one folder).
The log file should be saved in the same folder (so in our new folder structure,
every folder correspond to one experiment: server submission, log file and 
config file.)

This function allows to specify a config file at the start which is used as a default file.

2. A config parser program which based on a config file initiates the program. This should 
as general as possible since it initiates all programs and tasks in this project.
'''

#Load data frame of hyperparameters:
VALID_PATH='hyperparameters/hyperparameters.csv'
DEFAULT_PATH='hyperparameters/config_sota.ini'
DF_HP = pd.read_csv(VALID_PATH,index_col=0)
print(DF_HP.head())

def get_valid_values(name):
    valid_values=DF_HP.loc[name,:][4:].values
    valid_values = valid_values[~pd.isnull(valid_values)]
    return(valid_values)

def set_if_allowed(name,input_):
    if DF_HP.loc[name,"Type"]=='string':
        input_=str(input_)
    elif DF_HP.loc[name,"Type"]=='int':
        input_=int(input_)
    elif DF_HP.loc[name,"Type"]=='float':  
        input_=float(input_)
    else: 
        sys.exit("Error in hyperparameters: Unknown type.")

    if DF_HP.loc[name,"Space"]=='values':
        valid_values=get_valid_values(name)
        #Control whether input_ is a valid value:
        if input_ in valid_values:
            return(input_)
        
        #Otherwise start again:
        else:
            print("Invalid value for "+name+".")
            print("Possible values are: ", valid_values)
            return(set_hp(name))

    elif DF_HP.loc[name,"Space"]=='range':

        #Control whether lower and upper bound holds:
        lower_bound=(input_>=DF_HP.loc[name,"Min"]) or pd.isnull(DF_HP.loc[name,"Min"])
        upper_bound=(input_<=DF_HP.loc[name,"Max"]) or pd.isnull(DF_HP.loc[name,"Max"])
        if lower_bound and upper_bound:
            return(input_)

        #Otherwise start again:
        else:
            print("Value out of range for "+name+".")
            print("Range is  [", DF_HP.loc[name,"Min"],",",DF_HP.loc[name,"Max"],"]")
            return(set_hp(name))
    elif DF_HP.loc[name,"Space"]=='any':
        return(input_)
    else: 
        sys.exit("Error in hyperparameters file: unknown space.")


def set_hp(name,section=None,template=None):
    #Set prompt including default from template:
    prompt=name+": "
    if template is not None and section is not None:
        default=str(template_config[section][name])
        prompt=prompt+" || default: "+default+" || "
    
    #Receive input:
    input_= input(prompt)

    #If no input given, set input to template:
    if len(input_)==0 and template is not None and section is not None:
        input_=template_config[section][name]
    
    #If length zero, try again:
    if len(input_)==0:
        print("Length=0 invalid.")
        return(set_hp(name))
    else: 
        return(set_if_allowed(name,input_))

def file_if_exists(name):
    path = input (name+" || Default: "+DEFAULT_PATH+"|| : ")
    if osp.isfile(path) and len(path)>0:
        return(path)
    elif len(path)>0:
        print("File does not exist.")
        return(file_if_exists(name))
    else: 
        return DEFAULT_PATH

template_path = file_if_exists('template')
if template_path is not None:
    template_config=configparser.ConfigParser()
    template_config.read(template_path)
else: 
    template_config=None

#Get config parser:
config = configparser.ConfigParser()


#1. Specify the data set, the task we want to apply
#and the method we want to use.
config['experiment']={}
exp_config=config['experiment']
for key in template_config['experiment'].keys():
    exp_config[key]=str(set_hp(key,'experiment',template_config))

#Set all hyperparameters needed for specific method:
hp_sec=str(exp_config['method']+'_'+exp_config['task']+'_'+exp_config['data'])
config[hp_sec]={}
hp_sec_config=config[hp_sec]
for key in template_config[hp_sec].keys():
    hp_sec_config[key]=str(set_hp(key,hp_sec,template_config))

#Set all experiment hyperparameters:
if exp_config['save_config']=='yes':
    config['record']={}
    record_config=config['record']
    for key in template_config['record'].keys():
        record_config[key]=str(set_hp(key,'record',template_config))

    direct = osp.join(exp_config['parent_directory'],
                        hp_sec_config['folder'],
                        record_config['experiment_name'])
    if not osp.exists(direct):
        os.makedirs(direct)
    else: 
        print("Directory already exists.")

    date_string=datetime.today().strftime('%Y%m%d_%H%M')
    config_file_name=osp.join(direct,record_config['experiment_name']+date_string+'.ini')
    with open(config_file_name, 'w') as configfile:
        config.write(configfile)
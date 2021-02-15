import json
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
VALID_PATH='hps/valid_hps.csv'
DEFAULT_PATH='hps/sota_hps.json'
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


def set_hp(name,default_val=None):
    #Set prompt including default from template:
    prompt=name+": "
    if  default_val is not None:
        default=str(default_val)
        prompt=prompt+" || default: "+default+" || "

    #Receive input:
    input_= input(prompt)

    #If no input given, set input to template:
    if len(input_)==0 and default_val is not None:
        input_=default_val
        return(set_if_allowed(name,input_))
    #If length zero, try again:
    elif len(input_)==0:
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
    with open(template_path, "r") as read_file:
        temp_data = json.load(read_file)
else: 
    temp_data=None

#Get config parser:
config_data={}

#1. Specify the data set, the task we want to apply
#and the method we want to use.
config_data['experiment']={}
exp_config=config_data['experiment']
for key in temp_data['experiment'].keys():
    exp_config[key]=set_hp(key,temp_data['experiment'][key])

#Set all hyperparameters needed for specific method:
method=exp_config['method']
task=exp_config['task']
data=exp_config['data']

#Set empty dictionary:
config_data[method]={}
config_data[method][task]={}
config_data[method][task][data]={}

setup_config=config_data[method][task][data]

hps_config=setup_config['hps']={} 
default_hps=temp_data[method][task][data]['hps']

for key in default_hps.keys():
    hps_config[key]=set_hp(key,default_hps[key])

comp_config=setup_config['hps']={} 
default_comp=temp_data[method][task][data]['computing']

for key in default_comp.keys():
    comp_config[key]=set_hp(key,default_comp[key])

#Set all experiment hyperparameters:
if exp_config['save_config']=='yes':
    config_data['record']={}
    record_config=config_data['record']

    for key in temp_data['record'].keys():
        record_config[key]=set_hp(key,temp_data['record'][key])

    direct = osp.join(exp_config['parent_directory'],
                        comp_config['folder'],
                        record_config['experiment_name'])
    if not osp.exists(direct):
        os.makedirs(direct)
    else: 
        print("Directory already exists.")

    date_string=datetime.today().strftime('%Y%m%d_%H%M')
    config_file_name=osp.join(direct,record_config['experiment_name']+date_string+'.json')
    with open(config_file_name, "w") as configfile:
        json.dump(config_data,configfile,indent=2)
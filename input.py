import json
import sys 
import pandas as pd
import numpy as np
import os.path as osp
import os
from datetime import datetime
from bmrc.write_bmrc_file import write_bmrc_file

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


#Load data frame of valid hyperparameters:
VALID_PAR_PATH='hps/valid_hps.csv'

#Load data frame of state-of-the-art hyperparameters:
DEFAULT_PAR_PATH='hps/sota_hps.json'
df_hp = pd.read_csv(VALID_PAR_PATH,index_col=0)

def get_valid_values(name):
    '''
    Input: 
        name - string - name of parameter to get valid values from
    '''
    valid_values=df_hp.loc[name,:][4:].values
    valid_values = valid_values[~pd.isnull(valid_values)]
    
    if df_hp.loc[name,"Type"]=='string':
        valid_values=[str(value) for value in valid_values]
    elif df_hp.loc[name,"Type"]=='int':
        valid_values=[int(value) for value in valid_values]
    elif df_hp.loc[name,"Type"]=='float':  
        valid_values=[float(value) for value in valid_values]
    elif df_hp.loc[name,"Type"]=='bool':  
        valid_values=[bool(value) for value in valid_values]
    elif df_hp.loc[name,"Type"]=='int_list':   
        sys.exit("Valid values for lists are not possible so far.")
    elif df_hp.loc[name,"Type"]=='float_list':   
        sys.exit("Valid values for lists are not possible so far.")
    else: 
        sys.exit("Error in hyperparameters: Unknown type.")
    return(valid_values)

def set_if_allowed(name,input_):
    
    if df_hp.loc[name,"Type"]=='string':
        input_=str(input_)
    elif df_hp.loc[name,"Type"]=='int':
        input_=int(input_)
    elif df_hp.loc[name,"Type"]=='float':  
        input_=float(input_)
    elif df_hp.loc[name,"Type"]=='bool':  
        input_=bool(input_)
    elif df_hp.loc[name,"Type"]=='int_list':  

        print(input_)
        print(type(input_))
        input_=[int(val) for val in list(input_)]

    elif df_hp.loc[name,"Type"]=='float_list':  
        
        print(input_)
        print(type(input_))
        input_=[float(val) for val in list(input_)]
    
    else: 
        sys.exit("Error in hyperparameters: Unknown type.")

    if df_hp.loc[name,"Space"]=='values':
        valid_values=get_valid_values(name)
        #Control whether input_ is a valid value:
        if input_ in valid_values:
            return(input_)
        
        #Otherwise start again:
        else:
            print("Invalid value for "+name+".")
            print("Possible values are: ", valid_values)
            return(set_hp(name))

    elif df_hp.loc[name,"Space"]=='range':

        #Control whether lower and upper bound holds:
        lower_bound=(input_>=df_hp.loc[name,"Min"]) or pd.isnull(df_hp.loc[name,"Min"])
        upper_bound=(input_<=df_hp.loc[name,"Max"]) or pd.isnull(df_hp.loc[name,"Max"])
        if lower_bound and upper_bound:
            return(input_)

        #Otherwise start again:
        else:
            print("Value out of range for "+name+".")
            print("Range is  [", df_hp.loc[name,"Min"],",",df_hp.loc[name,"Max"],"]")
            return(set_hp(name))
    elif df_hp.loc[name,"Space"]=='any' or df_hp.loc[name,"Space"]=='bool':
        return(input_)
    else: 
        sys.exit("Error in hyperparameters file for %s. Unknown space: %s"%(name,df_hp.loc[name,"Space"]))


def set_hp(name,default_val=None):
    #Set prompt including default from template:
    prompt=name+": "
    if  default_val is not None:
        default=str(default_val)
        prompt=prompt+" || default: "+default+" || "

    #Receive input:
    if df_hp.loc[name,"Type"]=='int_list':  
        n=6
        input_ = list(map(int,input(prompt).strip().split()))[:n] 
    else:
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
    path = input (name+" || Default: "+DEFAULT_PAR_PATH+"|| : ")
    if osp.isfile(path) and len(path)>0:
        return(path)
    elif len(path)>0:
        print("File does not exist.")
        return(file_if_exists(name))
    else: 
        return DEFAULT_PAR_PATH

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

comp_config=setup_config['computing']={} 
default_comp=temp_data[method][task][data]['computing']

for key in default_comp.keys():
    comp_config[key]=set_hp(key,default_comp[key])

#Set all experiment hyperparameters:
if exp_config['save_config']=='yes':
    config_data['record']={}
    record_config=config_data['record']

    for key in temp_data['record'].keys():
        if key=='experiment_name':
            record_config[key]=set_hp(key,None)
        else:
            record_config[key]=set_hp(key,temp_data['record'][key])

    direct = osp.join(exp_config['parent_directory'],
                        comp_config['folder'],
                        record_config['experiment_name'])
    if not osp.exists(direct):
        os.makedirs(direct)
    else: 
        print("Directory already exists.")

    date_string=datetime.today().strftime('%Y%m%d_%H%M')
    json_filename=record_config['experiment_name']+date_string+'.json'

    json_filepath=osp.join(direct,json_filename)

    with open(json_filepath, "w") as configfile:
        json.dump(config_data,configfile,indent=2)
    
    if exp_config['save_server']=='yes':
        server_filename=record_config['experiment_name']+date_string+'.sh'
        log_filename=record_config['experiment_name']+date_string+'.log'

        server_filepath=osp.join(direct,server_filename)
        log_filepath=osp.join(direct,log_filename)
        write_bmrc_file(comp_config['queue'],comp_config['n_gpus'],json_filepath,log_filepath,server_filepath)

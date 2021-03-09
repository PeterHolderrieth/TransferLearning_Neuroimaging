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
DF_HP = pd.read_csv(VALID_PAR_PATH,index_col=0)

#Load data frame of state-of-the-art hyperparameters:
DEFAULT_PAR_PATH='hps/sota_hps.json'

def input_bool(string):
    x=input(string)
    if len(x)==0:
        return "EMPTY"
    if x=="True":
        return True
    elif x=="False":
        return False
    else: 
        print("Possible values are 'True' or 'False'.")
        return input_bool(string)

def input_float(string):
    x=input(string)
    if len(x)==0:
        return "EMPTY"
    try:
        x=float(x)
        return(x)
    except ValueError:
        print("Not a float")
        return(input_float(string))

def input_string(string):
    x=input(string)
    if len(x)==0:
        return "EMPTY"
    try:
        x=str(x)
        return(x)
    except ValueError:
        print("Not a string")
        return(input_float(string))

def input_int(string):
    x=input(string)
    if len(x)==0:
        return "EMPTY"
    try:     
        x=float(x)
        if x.is_integer():
            return(int(x))
        else: 
            print("An int, not a float, is required.")
            return(input_int(string))
    except ValueError:
        print("Not an int.")
        return(input_int(string))

def input_list(string,type_):
    print(string)
    n=input_int("Number of elements: ")
    if n=="EMPTY":
        return "EMPTY"
    else:
        my_list=[]
        for it in range(n):
            x=my_input("El %2d: "%it, type_)
            my_list.append(x)
        return(my_list)

def my_input(string,type_):
    if type_=='int':
        return input_int(string)
    elif type_=='float':
        return input_float(string)
    elif type_=='bool':
        return input_bool(string)
    elif type_=='string':
        return input_string(string)
    elif type_=='int_list':
        return input_list(string,"int")
    elif type_=='float_list':
        return input_list(string,"float")
    elif type_=='bool_list':
        return input_list(string,"bool")
    elif type_=='string_list':
        return input_list(string,"string")
    else: 
        sys.exit("Unknown input type.")

def get_valid_values(name):
    '''
    Input: 
        name - string - name of parameter to get valid values from
    '''
    valid_values=DF_HP.loc[name,:][4:].values
    valid_values = valid_values[~pd.isnull(valid_values)]
    
    if DF_HP.loc[name,"Type"]=='string':
        valid_values=[str(value) for value in valid_values]
    elif DF_HP.loc[name,"Type"]=='int':
        valid_values=[int(value) for value in valid_values]
    elif DF_HP.loc[name,"Type"]=='float':  
        valid_values=[float(value) for value in valid_values]
    elif DF_HP.loc[name,"Type"]=='bool':  
        valid_values=[bool(value) for value in valid_values]
    elif DF_HP.loc[name,"Type"]=='int_list':   
        sys.exit("Valid values for lists are not possible so far.")
    elif DF_HP.loc[name,"Type"]=='float_list':   
        sys.exit("Valid values for lists are not possible so far.")
    else: 
        sys.exit("Error in hyperparameters: Unknown type.")
    return(valid_values)

def set_if_allowed(name,input_):
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
    elif DF_HP.loc[name,"Space"]=='any' or DF_HP.loc[name,"Space"]=='bool':
        return(input_)
    else: 
        sys.exit("Error in hyperparameters file for %s. Unknown space: %s"%(name,DF_HP.loc[name,"Space"]))


def set_hp(name,default_val=None):

    #Set prompt including default from template
    prompt=name+": "
    if  default_val is not None:
        default=str(default_val)
        prompt=prompt+" || default: "+default+" || "

    #Get type:
    type_=DF_HP.loc[name,"Type"]
    
    #Get input:
    input_= my_input(prompt,type_)

    #If no input given, set input to template
    if input_=="EMPTY" and default_val is not None:
        input_=default_val
        return(set_if_allowed(name,input_))

    #If no input and no template is given, try again
    elif input_=="EMPTY":
        print("Length=0 invalid if no default is given.")
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


for el in exp_config.keys():
    print(exp_config[el])
    print(type(exp_config[el]))

#Set all experiment hyperparameters:
if exp_config['save_config']:
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
    
    if exp_config['save_server']:
        server_filename=record_config['experiment_name']+date_string+'.sh'
        log_filename=record_config['experiment_name']+date_string+'.log'

        server_filepath=osp.join(direct,server_filename)
        log_filepath=osp.join(direct,log_filename)
        write_bmrc_file(comp_config['queue'],comp_config['n_gpus'],json_filepath,log_filepath,server_filepath)

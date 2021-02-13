import configparser
import sys 
import pandas as pd
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
'''
TASK_LIST=['age','sex']
METHOD_LIST=['dl_scratch','finetune_final_layer','regression']

def set_if_allowed(value,allowed_values):
    if value in allowed_values:
        return(value)
    else: 
        sys.exit("Invalid value ", value, " for ", allowed_values.__name__) 

class Configurations(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = configparser.ConfigParser(config_path)
        self.info=self.config['info']
        self.tasks=set_if_allowed(self.config['experiment']['task'],TASK_LIST)

config=Configurations('example.ini')
'''
#Load data frame of hyperparameters:
PATH='hyperparameters/hyperparameters.csv'
DF_HP = pd.read_csv(PATH,index_col=0)
print(DF_HP)


'''
a=4
input_ = int(input("Enter the inputs. We suggest: "+ str(a)+".  || ") or a)
print(input_)
test_text = input ("Enter a number: ")
print(type(test_text))
print(test_text)
test_number = input ("Enter another number: ")

print("Directory: ")

template_path = input ("Template: ")

print("Template: ")
'''


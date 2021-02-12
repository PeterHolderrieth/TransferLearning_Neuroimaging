import configparser
import sys 

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
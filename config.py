import configparser
import sys 

class Configurations(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.info=self.config['info']
        self.tasks=self.config['experiment']['task']

configurations=Configurations('example.ini')

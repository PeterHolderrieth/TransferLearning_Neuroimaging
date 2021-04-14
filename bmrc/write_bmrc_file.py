#This function creates a server job from instructions.
#It is made for the BMRC server at the University of Oxford.

def write_bmrc_file(queue,n_gpus,json_filepath,log_filepath,server_filepath):
    '''
    queue - string - queue to submit file to
    n_gpus - int - number of GPUs
    json_filepath - string - file path to json file 
    log_filepath - string - file path to log file path where to save results of the jobs
    server_filepath - string - file path where to save resuls
    '''
    #Read template file:
    f = open("bmrc/general_template.sh", "r")
    contents = f.readlines()
    f.close()
    
    #Insert appropriate input:
    contents[3]=contents[3]%queue
    contents[5]=contents[5]%str(n_gpus)
    contents[6]=contents[6]%str(n_gpus)
    contents[10]=contents[10]%log_filepath
    contents[26]=contents[26]%json_filepath

    #Save file:
    f = open(server_filepath, "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()



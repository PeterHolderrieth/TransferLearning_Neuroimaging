def write_bmrc_file(queue,n_gpus,json_filepath,log_filepath,server_filepath):
    f = open("bmrc/general_template.sh", "r")
    contents = f.readlines()
    f.close()
    
    contents[3]=contents[3]%queue
    contents[5]=contents[5]%str(n_gpus)
    contents[6]=contents[6]%str(n_gpus)
    contents[10]=contents[10]%log_filepath
    contents[26]=contents[26]%json_filepath

    f = open(server_filepath, "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()



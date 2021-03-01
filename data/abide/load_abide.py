# WHY NOT SHUFFLE FOR THE TRAIN LOADERS?
# WHAT IS THE DIFFERENCE BETWEEN TEST SET 1 and TEST SET 2?
import sys 
import torch
import numpy as np
import nibabel as nib
import os
import os.path as osp
import pandas as pd 

sys.path.append('../../')
from data.dataset import give_mri_data

def give_abide_data(data_type,batch_size=1,num_workers=1,shuffle=True,debug=False,preprocessing='full', task='age',share=None):
    
    #Get the directory of the data_type:
    DIR = '/gpfs3/well/win-fmrib-analysis/users/lhw539/abide/'
    DIR_IDs=osp.join(DIR, 'info/') 
    default_name=DIR_IDs
    if debug:
        default_name+='debug_'
    
    # Load files:
    if data_type=='train':
        fp_ = default_name+'abide_train.csv'
    elif data_type=='val':
        fp_ = default_name+'abide_val.csv'
    elif data_type=='test0':
        fp_ = default_name+'abide_test.csv'
    else: 
        sys.exit("Unknown data type.")
    
    df = pd.read_csv(fp_)
    
    fp_list=df["T1_brain_lin"].to_list()

    if task=='age':
        label_list=list([age_,] for age_ in df["Age"].to_list())
    elif task=='sex':
        label_list=list([sex_,] for sex_ in df["Sex"].to_list())
    else: 
        sys.exit("Unknown task.")

    if share is not None: 
        n_total=len(fp_list)
        n_samples=int(np.round(share*n_total))
        inds=torch.randperm(n=n_total)[:n_samples].tolist()
        fp_list=[fp_list[ind] for ind in inds]
        label_list=[label_list[ind] for ind in inds]
    
    if  debug:    
        if share is None or share>0.99999:
            print("Loading ABIDE %5s debug data."%data_type)
        else: 
            print("Loading share %.2f ABIDE %5s debug data."%(share,data_type))
    else: 
        if share is None or share>0.99999:
            print("Loading ABIDE %5s data."%data_type)
        else: 
            print("Loading share %.2f %5s data."%(share,data_type))

    
    return(give_mri_data(fp_list=fp_list,label_list=label_list,data_type=data_type,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,preprocessing=preprocessing))





# WHY NOT SHUFFLE FOR THE TRAIN LOADERS?
# WHAT IS THE DIFFERENCE BETWEEN TEST SET 1 and TEST SET 2?
import sys 
import torch
import numpy as np
import nibabel as nib
import os
import os.path as osp
import pandas as pd 

from dataset import construct_preprocessing 
from dataset import MRIDataset

def give_oasis_data(data_type,batch_size=1,num_workers=1,shuffle=True,debug=False):

    
    #Construct preprocessing functions:
    ps = construct_preprocessing({'method': 'pixel_shift',
                                        'x_shift': 2,
                                        'y_shift': 2,
                                        'z_shift': 2})
    mr = construct_preprocessing({'method': 'mirror',
                                        'probability': 0.5})
    avg = construct_preprocessing({'method': 'average'})
    crop = construct_preprocessing({'method': 'crop',
                                            'nx': 160,
                                            'ny': 192,
                                            'nz': 160})
	

    #Get the directory of the data_type:
    DIR = '/gpfs3/well/win-fmrib-analysis/users/lhw539/oasis3/'
    DIR_IDs=osp.join(DIR, 'oasis3_info/') 
    default_name=DIR_IDs
    if debug:
        default_name=DIR_IDs+'debug_'

    # Load files:
    if data_type=='train':
        fp_ = default_name+'session_train.csv'
    elif data_type=='val':
        fp_ = default_name+'session_val.csv'
    elif data_type=='test0':
        fp_ = default_name+'session_test0.csv'
    elif data_type=='test1':
        fp_ = default_name+'session_test1.csv'
    else: 
        sys.exit("Unknown data type.")
    
    df_session = pd.read_csv(fp_)

    #Load T1 file path values - in the special ase of test1, we use ????:
    if data_type=='test1':
        fp_ = osp.join(DIR_IDs, 'subject_test1.csv')
        if debug:
            fp_=osp.join('debug_',fp_)
        df_subject_test1 = pd.read_csv(fp_)
        fp_list = list(df_subject_test1.max_cdr_mri_T1_path.values)
    else: 
        fp_list = list(df_session.T1_path.values)
    
    #Get list of labels:
    label_list = list([age_, ] for age_ in df_session.AgeBasedOnClinicalData.values)

    if data_type=='train':
        data_set = MRIDataset(fp_list, label_list, [avg,crop])#mr, ps, avg, crop])
    else: 
        data_set = MRIDataset(fp_list, label_list, [avg,crop])#avg, crop])

    #Return data loader:
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True
    )
    print("Succesfully loaded OASIS %5s data."%data_type)

    return(data_set,data_loader)


# WHY NOT SHUFFLE FOR THE TRAIN LOADERS?
# WHAT IS THE DIFFERENCE BETWEEN TEST SET 1 and TEST SET 2?

import nibabel as nib
import sys 
from dataset import construct_preprocessing 
from dataset import MRIDataset
import torch
import pandas as pd 
import numpy as np
import os
import os.path as osp

#data_folder="/well/win-fmrib-analysis/users/lhw539/oasis3/data/"
#example_file="OAS30869_MR_d1691_anat_T1_brain_to_MNI_lin.nii.gz"
#img = nib.load(data_folder+example_file)
#data=img.get_fdata()
#print(type(data))
#print(data.shape)

def give_oasis_data(data_type,batch_size=1,num_workers=1,shuffle=True):


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

    # Load files:
    if data_type=='train':
        fp_ = osp.join(DIR_IDs, 'session_train.csv')
    elif data_type=='val':
        fp_ = osp.join(DIR_IDs, 'session_val.csv')
    elif data_type=='test0':
        fp_ = osp.join(DIR_IDs, 'session_test0.csv')
    elif data_type=='test1':
        fp_ = osp.join(DIR_IDs, 'session_test1.csv')

    df_session = pd.read_csv(fp_)

    #Load T1 file path values - in the special ase of test1, we use ????:
    if data_type=='test1':

        fp_list = list(df_subject_test1.max_cdr_mri_T1_path.values)
    else: 
        fp_list = list(df_session.T1_path.values)
    
    #Get list of labels:
    label_list = list([age_, ] for age_ in df_session.AgeBasedOnClinicalData.values)

    if data_type=='train':
        data_set = MRIDataset(fp_list, label_list, [mr, ps, avg, crop])
    else: 
        data_set = MRIDataset(fp_list, label_list, [avg, crop])

    #Return data loader:
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True
    )

    return(data_set,data_loader)

subjects_train=np.unique(df_session_train.Subject.values)
subjects_val=np.unique(df_session_val.Subject.values)
subjects_test0=np.unique(df_session_test0.Subject.values)
subjects_test1=np.unique(df_session_test1.Subject.values)
print(subjects_val.shape)
print(subjects_test0.shape)
print(subjects_test1.shape)
print(np.intersect1d(subjects_train,subjects_val).shape)
print(np.intersect1d(subjects_val,subjects_test0).shape)
print(np.intersect1d(subjects_train,subjects_test0).shape)
print(np.intersect1d(subjects_test0,subjects_test1).shape)
print(np.intersect1d(subjects_test1,subjects_train).shape)
print(np.intersect1d(subjects_test1,subjects_val).shape)
    
'''    
    if dataset='train':
        fp_ = osp.join(DIR_IDs, 'subject_train.csv')
    
    df_subject_train = pd.read_csv(fp_)
    fp_ = osp.join(DIR_IDs, 'subject_val.csv')
    df_subject_val = pd.read_csv(fp_)
    fp_ = osp.join(DIR_IDs, 'subject_test0.csv')
    df_subject_test0 = pd.read_csv(fp_)
    fp_ = osp.join(DIR_IDs, 'subject_test1.csv')
    df_subject_test1 = pd.read_csv(fp_)
'''
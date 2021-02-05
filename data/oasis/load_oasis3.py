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
from dataset import construct_preprocessing 
from dataset import MRIDataset

def give_oasis_data(data_type,batch_size=1,num_workers=1,shuffle=True,debug=False,preprocessing='full', task='age'):

    
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
	
    #Pick a list of preprocessing functions:
    if preprocessing=='full':
        preproc_train=[avg,crop,ps,mr]
        preproc_val=[avg,crop]

    elif preprocessing=='min':
        preproc_train=[avg,crop]
        preproc_val=[avg,crop]
 
    elif preprocessing=='none':
        preproc_train=[]
        preproc_val=[]
 
    else:
        sys.exit("Unknown preprocessing combination.")

    #Get the directory of the data_type:
    DIR = '/gpfs3/well/win-fmrib-analysis/users/lhw539/oasis3/'
    DIR_IDs=osp.join(DIR, 'oasis3_info/') 
    default_name=DIR_IDs
    if debug:
        default_name+='debug_'

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

    #Load T1 file path values. In the special ase of test1 (dementia group),
    #we use only the scans with highest CDR score.
    if data_type=='test1':
        fp_ = osp.join(DIR_IDs, 'subject_test1.csv')
        if debug:
            fp_=osp.join('debug_',fp_)
        df_subject_test1 = pd.read_csv(fp_)
        fp_list = list(df_subject_test1.max_cdr_mri_T1_path.values)
    else: 
        fp_list = list(df_session.T1_path.values)
    
    
    if task=='age':
        label_list = list([age_, ] for age_ in df_session.AgeBasedOnClinicalData.values)
    
    elif task=='sex':
        #Get subject info:
        subject_df=pd.read_csv(DIR_IDs+'subject_info.csv')
        #Extract subject and sex info. Set subject as index:
        subject_sex=subject_df[["Subject","Sex"]].set_index("Subject")
        #Extract labels:
        label_list=subject_sex.loc[df_session.Subject.values,:]
    else: 
        sys.exit("Unknown task.")

    if data_type=='train':
        data_set = MRIDataset(fp_list, label_list, preproc_train)
    else: 
        data_set = MRIDataset(fp_list, label_list, preproc_val)

    
    batch_size_=min(data_set._len,batch_size)

    #Return data loader:
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size_,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True
    )

    #Print information:
    if  debug:    
        print("Succesfully loaded OASIS %5s debug data."%data_type)
    else: 
        print("Succesfully loaded OASIS %5s data."%data_type)
    return(data_set,data_loader)



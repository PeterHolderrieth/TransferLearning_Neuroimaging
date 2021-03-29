

import sys 
import torch
import numpy as np
import nibabel as nib
import os
import os.path as osp
import pandas as pd 

sys.path.append('../../')
from data.dataset import give_mri_data



def give_ukb_data(data_type,batch_size=1,num_workers=1,shuffle=True,debug=False,preprocessing='full', task='age',share=None,balance=None):

    DIR='/gpfs3/well/win-fmrib-analysis/users/lhw539/ukb/'
    
    #Get the directory of the data_type:
    if data_type=='full':
        fp_= DIR+'full_info.csv'

    elif data_type=='test' or data_type=='val' or data_type=='train':
        fp_= DIR+'test_info.csv'

    else:
        sys.exit("Unknown data type.")

    df_session = pd.read_csv(fp_)
    fp_list=df_session.T1_brain_to_MNI_linear.to_list()
    
    if task=='age':
         label_list = list([age_, ] for age_ in df_session.Age.values)
    
    elif task=='sex':

        label_list=list([sex_, ] for sex_ in df_session.Sex.values)        
    
    else: 
        sys.exit("Unknown task.")

    n_total=len(fp_list)

    if debug:
        share=10/n_total

    if share is not None: 
        n_samples=int(np.round(share*n_total))
        inds=torch.randperm(n=n_total)[:n_samples].tolist()
        fp_list=[fp_list[ind] for ind in inds]
        label_list=[label_list[ind] for ind in inds]

    if share is None: 
        print("Loading UKB %5s data."%data_type)
    else: 
        print("Loading share %.2f of UKB %5s data."%(share,data_type))

    
    return(give_mri_data(fp_list=fp_list,label_list=label_list,data_type=data_type,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,preprocessing=preprocessing))
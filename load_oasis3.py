## WHY NOT SHUFFLE FOR THE TRAIN LOADERS?
#WHAT IS THE DIFFERENCE BETWEEN ls
import nibabel as nib
import sys 
from dataset import construct_preprocessing 
from dataset import MRIDataset
import torch
import pandas as pd 
import os
import os.path as osp

#data_folder="/well/win-fmrib-analysis/users/lhw539/oasis3/data/"
#example_file="OAS30869_MR_d1691_anat_T1_brain_to_MNI_lin.nii.gz"
#img = nib.load(data_folder+example_file)
#data=img.get_fdata()
#print(type(data))
#print(data.shape)

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

batch_size=10
num_workers=4

#Get the directory of the dataset:
DIR = '/gpfs3/well/win-fmrib-analysis/users/lhw539/oasis3/'
DIR_IDs=osp.join(DIR, 'oasis3_info/') 

# Load files:
fp_ = osp.join(DIR_IDs, 'subject_train.csv')
df_subject_train = pd.read_csv(fp_)
fp_ = osp.join(DIR_IDs, 'subject_val.csv')
df_subject_val = pd.read_csv(fp_)
fp_ = osp.join(DIR_IDs, 'subject_test0.csv')
df_subject_test0 = pd.read_csv(fp_)
fp_ = osp.join(DIR_IDs, 'subject_test1.csv')
df_subject_test1 = pd.read_csv(fp_)

fp_ = osp.join(DIR_IDs, 'session_train.csv')
df_session_train = pd.read_csv(fp_)
fp_ = osp.join(DIR_IDs, 'session_val.csv')
df_session_val = pd.read_csv(fp_)
fp_ = osp.join(DIR_IDs, 'session_test0.csv')
df_session_test0 = pd.read_csv(fp_)
fp_ = osp.join(DIR_IDs, 'session_test1.csv')
df_session_test1 = pd.read_csv(fp_)

#Load the datasets:
fp_list = list(df_session_train.T1_path.values)
label_list = list([age_, ] for age_ in df_session_train.AgeBasedOnClinicalData.values)
dataset_train = MRIDataset(fp_list, label_list, [mr, ps, avg, crop])

fp_list = list(df_session_val.T1_path.values)
label_list = list([age_, ] for age_ in df_session_val.AgeBasedOnClinicalData.values)
dataset_val = MRIDataset(fp_list, label_list, [mr, ps, avg, crop])

fp_list = list(df_session_test0.T1_path.values)
label_list = list([age_, ] for age_ in df_session_test0.AgeBasedOnClinicalData.values)
dataset_test0 = MRIDataset(fp_list, label_list, [avg, crop])

fp_list = list(df_subject_test1.max_cdr_mri_T1_path.values)
label_list = list([age_, ] for age_ in df_session_test1.AgeBasedOnClinicalData.values)
dataset_test1 = MRIDataset(fp_list, label_list, [avg, crop])


train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)
test_loader0 = torch.utils.data.DataLoader(
    dataset_test0,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)
test_loader1 = torch.utils.data.DataLoader(
    dataset_test1,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)
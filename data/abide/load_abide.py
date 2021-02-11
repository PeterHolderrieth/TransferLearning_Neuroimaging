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
from dataset import give_mri_data

#Load abide info file:
#path='/gpfs3/well/win-fmrib-analysis/users/lhw539/abide/info/abide_info_clean_strict.csv'
path_info='/well/win-biobank/users/jdo465/datasets/abide/abide_info_clean_strict.csv'
df_abide = pd.read_csv(path_info)
print(df_abide.head())

name=str(int(df_abide.iloc[0,0]))
print(name)

#Go over subject numbers: 
subject_numbers=df_abide.iloc[:,0]
subjects,counts=np.unique(subject_numbers,return_counts=True)
print(np.max(counts),np.max(counts))

print("Number of subject numbers: ", subjects.shape)


df_abide["path"]=''
DIR='/well/win-biobank/users/jdo465/datasets/abide/'
for it in range(subject_numbers.shape[0]):
    abide1_name='abide1/mprage_00'+name+'.anat/T1_to_MNI_lin.nii.gz'
    abide2_name='abide2/anat_'+name+'.anat/T1_to_MNI_lin.nii.gz'
    if os.path.isfile(DIR+abide1_name):
        filename=DIR+abide1_name
    elif os.path.isfile(DIR+abide2_name):
        filename=DIR+abide2_name
    else:
        sys.exit("File unknown.")
    df_abide.at[it,"path"]=filename

x=np.array([len(string) for string in df_abide.iloc[:,6]])
print(np.unique(x,return_counts=True))


#For later use:
'''
def give_oasis_data(data_type,batch_size=1,num_workers=1,shuffle=True,debug=False,preprocessing='full', task='age'):
    
    #Get the directory of the data_type:
    DIR = '/gpfs3/well/win-fmrib-analysis/users/lhw539/abide/'
    DIR_IDs=osp.join(DIR, 'info/') 
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
        label_list=list([sex_, ] for sex_ in subject_sex.loc[df_session.Subject.values,:].values)
    else: 
        sys.exit("Unknown task.")
    
    if  debug:    
        print("Loading Abide %5s debug data."%data_type)
    else: 
        print("Loading loaded Abide %5s data."%data_type)

    return(give_mri_data(fp_list=fp_list,label_list=label_list,data_type=data_type,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,preprocessing=preprocessing))

'''


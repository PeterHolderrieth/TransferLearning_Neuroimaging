# WHY NOT SHUFFLE FOR THE TRAIN LOADERS?
# WHAT IS THE DIFFERENCE BETWEEN TEST SET 1 and TEST SET 2?
import sys 
import torch
import numpy as np
import nibabel as nib
import os
import os.path as osp
import pandas as pd 
from data.dataset import give_mri_data

def test_uniqueness_of_elements(arr):
    arr_unique,counts=np.unique(arr,return_counts=True)
    if np.max(counts)==np.min(counts) and np.min(counts)==1:
        return True 
    else: 
        return False

def split_data_frame(df,share_train,share_val):
    '''
    shares - list of ints - shares of the data set
    '''
    nrows=df.shape[0]
    n_train=int(np.round(nrows*share_train))
    n_val=int(np.round(nrows*share_val))
    n_test=nrows-n_train-n_val
    ind=np.random.permutation(nrows)
    ind_train=ind[:n_train]
    ind_val=ind[n_train:n_val]
    ind_test=ind[n_val:]

    return df.loc[ind_train],df.loc[ind_val],df.loc[ind_test]



#Load abide info file:
#path='/gpfs3/well/win-fmrib-analysis/users/lhw539/abide/info/abide_info_clean_strict.csv'
path_info='/gpfs3/well/win-fmrib-analysis/users/lhw539/abide/info/abide_info_clean_strict_with_filepath.csv'
df_abide = pd.read_csv(path_info,index_col=0)
print(df_abide.head())

subject_numbers=df_abide["No"].to_numpy()
if test_uniqueness_of_elements(subject_numbers):
    print("We have unique subject numbers")

print("Number of subjects: ", subject_numbers.shape)

filenames=df_abide["T1_brain_lin"].to_numpy()
if test_uniqueness_of_elements(filenames):
    print("We have unique filenames.")

age=df_abide["Age"].to_numpy()
min_age=np.min(age)
max_age=np.max(age)
std_age=np.std(age)
mean_age=np.mean(age)

print("Minimum: ", min_age)
print("Maximum: ", max_age)
print("Std age: ", std_age)
print("Mean age: ", mean_age)


sites=np.unique(df_abide["SiteNameCombine"].to_numpy())

share_train=0.4
share_valid=0.2
share_test=1-share_train-share_valid

for site in sites: 
    df_site=df_abide.loc[df_abide["SiteNameCombine"]==site]
    df_it_train,df_it_val,df_it_test=split_data_frame(df_site,share_train,share_valid)
    print(df_it_train.shape)
    print(df_it_val.shape)
    print(df_it_test.shape)



#x=np.array([len(string) for string in df_abide.iloc[:,6]])
#print(np.unique(x,return_counts=True))


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


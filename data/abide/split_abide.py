#This script splits the abide data into a train, validation and test set.
#The split leads to an (approximately) equal distribution of scanner sites and 
#age of subjects across data sets.

import sys 
import torch
import numpy as np
import nibabel as nib
import os
import os.path as osp
import pandas as pd 

#Set seed:
np.random.seed(301297)


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
    ind=np.random.permutation(nrows).astype(int)
    ind_train=ind[:n_train].tolist()
    ind_val=ind[n_train:(n_train+n_val)].tolist()
    ind_test=ind[(n_train+n_val):].tolist()
    
    return df.iloc[ind_train],df.iloc[ind_val],df.iloc[ind_test]

def print_summary(arr,name):
    min_arr=np.min(arr)
    max_arr=np.max(arr)
    std_arr=np.std(arr)
    mean_arr=np.mean(arr)

    print(name+'')
    print("Minimum: ", min_arr)
    print("Maximum: ", max_arr)
    print("Std arr: ", std_arr)
    print("Mean arr: ", mean_arr)
    print() 

#Load abide info file:
folder='/gpfs3/well/win-fmrib-analysis/users/lhw539/abide/'
path_info=folder+'info/abide_info_clean_strict_with_filepath.csv'

df_abide = pd.read_csv(path_info,index_col=0)

#Tesst whether all subject numbers are unique:
subject_numbers=df_abide["No"].to_numpy()
if test_uniqueness_of_elements(subject_numbers):
    print()
    print("We have unique subject numbers")

print("Number of subjects: ", subject_numbers.shape)

#Test whether all filenames are uniques:
filenames=df_abide["T1_brain_lin"].to_numpy()
if test_uniqueness_of_elements(filenames):
    print()
    print("We have unique filenames.")

#Get all unique sites:
sites=np.unique(df_abide["SiteNameCombine"].to_numpy())

#Split parameters of train, validation an test set:
share_train=0.4
share_valid=0.2
share_test=1-share_train-share_valid

#Initialize lists:
df_train_list=[]
df_val_list=[]
df_test_list=[]

#Go over all sites and split the subjects from one site into train, valid and test:
for site in sites: 
    df_site=df_abide.loc[df_abide["SiteNameCombine"]==site]
    df_it_train,df_it_val,df_it_test=split_data_frame(df_site,share_train,share_valid)
    df_train_list.append(df_it_train)
    df_val_list.append(df_it_val)
    df_test_list.append(df_it_test)

#Aggregrate train, validation and test subjects across scanner sites:
df_train=pd.concat(df_train_list)
df_val=pd.concat(df_val_list)
df_test=pd.concat(df_test_list)

#Print shape:
print()
print("Train shape: ", df_train.shape)
print("Validation shape: ", df_val.shape)
print("Test shape: ", df_test.shape)
print() 

#Print summary to see that the split is age-matched:
print_summary(df_abide["Age"],'Age')
print_summary(df_train["Age"],'Train Age')
print_summary(df_val["Age"],'Val Age')
print_summary(df_test["Age"],'Test Age')


#Save train, validation and test files:
df_train.to_csv(folder+'info/abide_train.csv')
df_val.to_csv(folder+'info/abide_val.csv')
df_test.to_csv(folder+'info/abide_test.csv')

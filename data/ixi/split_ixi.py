#This script splits the ixi data into a train, validation and test set.
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
np.random.seed(28061995)


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

#Load ixi info file:
folder='/gpfs3/well/win-fmrib-analysis/users/lhw539/ixi/'
path_info=folder+'ixi_clean_03012021.csv'

df_ixi = pd.read_csv(path_info,index_col=0)
print("Data shape: ", df_ixi.shape)

print("Variables: ", list(df_ixi.columns))

#Control whether AGE and Age are the same:
AGE=df_ixi["AGE"].to_numpy()
Age=df_ixi["AGE"].to_numpy()

if np.array_equal(AGE,Age):
    print("Age and AGE are the same.")


#Control what sex=0,sex=1 stands for:
orig=df_ixi["Sex"].to_numpy()
conv=(2-df_ixi.iloc[:,0]).to_numpy()

if np.array_equal(orig,conv):
    print("Sex=1 is male, sex=0 is female.")

#Test whether all filenames are uniques:
filenames=df_ixi["T1_brain"].to_numpy()
if test_uniqueness_of_elements(filenames):
    print()
    print("We have unique filenames.")

#Get all unique sites:
sites=np.unique(df_ixi["Site"].to_numpy())

print("Number of sites: ", sites.shape)
print("Sites: ", sites)

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
    df_site=df_ixi.loc[df_ixi["Site"]==site]
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
print_summary(df_ixi["Age"],'Age')
print_summary(df_train["Age"],'Train Age')
print_summary(df_val["Age"],'Val Age')
print_summary(df_test["Age"],'Test Age')


#Save train, validation and test files:
df_train.to_csv(folder+'ixi_train.csv')
df_val.to_csv(folder+'ixi_val.csv')
df_test.to_csv(folder+'ixi_test.csv')

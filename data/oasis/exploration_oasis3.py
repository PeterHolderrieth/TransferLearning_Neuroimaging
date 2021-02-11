#%%
import numpy as np
import matplotlib.pyplot as plt 
#%matplotlib inline
from load_oasis3 import give_oasis_data

def give_oasis_info(data_type,debug=False):
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
    return(df_session)

data_set,_=give_oasis_data('train',debug=False)
labels_train=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('val',debug=False)
labels_val=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('test0',debug=False)
labels_test0=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('test1',debug=False)
labels_test1=np.array(data_set.label_list).flatten()

col_list=['blue','orange','red','green']


def give_array_summary(x):
    dict_={
    'mean': x.mean(),
    'median': np.median(x),
    'min': x.min(),
    'max': x.max(),
    'std': x.std(),
    }
    return(dict_)

fig,ax=plt.subplots(ncols=2,nrows=2)
#print(give_array_summary(labels))
n_bins=20
ax[0,0].hist(labels_train,bins=n_bins,alpha=1,density=True,color=col_list[0])
ax[0,0].set_title("Train")

ax[0,1].hist(labels_val,bins=n_bins,alpha=1,density=True,color=col_list[1])
ax[0,1].set_title("Valid")

ax[1,0].hist(labels_test0,bins=n_bins,alpha=1,density=True,color=col_list[2])
ax[1,0].set_title("Test0")

ax[1,1].hist(labels_test1,bins=n_bins,alpha=1,density=True,color=col_list[3])
ax[1,1].set_title("Test1")

limits=[42,95]
ax[0,0].set_xlim(limits)
ax[0,1].set_xlim(limits)
ax[1,0].set_xlim(limits)
ax[1,1].set_xlim(limits)

fig.suptitle("Age distribution of samples (not subjects!).")
plt.savefig("exploration/plots/OASIS3_age_dist.png")
# %%
import os
import os.path as osp
import pandas as pd
# %%
df_train=give_oasis_info('train')
df_val=give_oasis_info('val')
df_test0=give_oasis_info('test0')
df_test1=give_oasis_info('test1')

# %%
#Print number of rows:
print(df_train.shape[0])
print(df_val.shape[0])
print(df_test0.shape[0])
print(df_test1.shape[0])
# %%
train_subjects,train_counts=np.unique(df_train.Subject.values,return_counts=True)
val_subjects,val_counts=np.unique(df_val.Subject.values,return_counts=True)
test0_subjects,test0_counts=np.unique(df_test0.Subject.values,return_counts=True)
test1_subjects,test1_counts=np.unique(df_test1.Subject.values,return_counts=True)

#%%
fig,ax=plt.subplots(ncols=2,nrows=1)
ax[0].bar(x=["Train","Valid","Test0","Test1"],
        height=[train_subjects.shape[0],val_subjects.shape[0],
        test0_subjects.shape[0],test1_subjects.shape[0]],color=col_list)
ax[0].set_title("Number of subjects.")
ax[1].bar(x=["Train","Valid","Test0","Test1"],
        height=[df_train.shape[0],df_val.shape[0],
        df_test0.shape[0],df_test1.shape[0]],color=col_list)
ax[1].set_title("Number of sessions (=samples).")
plt.savefig("exploration/plots/OASIS3_numbers.png")

# %%
#The distribution is quite similar:
fig, ax=plt.subplots(ncols=2,nrows=2)
ax[0,0].hist(train_counts,bins=[0.5,1.5,2.5,3.5,4.5,5.5],color=col_list[0])
ax[0,0].set_title("Train")
ax[0,1].hist(val_counts,bins=[0.5,1.5,2.5,3.5,4.5,5.5],color=col_list[1])
ax[0,1].set_title("Validation")
ax[1,0].hist(test0_counts,bins=[0.5,1.5,2.5,3.5,4.5,5.5],color=col_list[2])
ax[1,0].set_title("Test0")
ax[1,1].hist(test1_counts,bins=[0.5,1.5,2.5,3.5,4.5,5.5],color=col_list[3])
ax[1,1].set_title("Test1")
fig.suptitle("Number of sessions per subject")
plt.savefig("exploration/plots/OASIS3_sessions_per_subject.png")

# %%
#All intersections are empty:
print(np.intersect1d(train_subjects,val_subjects))
print(np.intersect1d(train_subjects,test0_subjects))
print(np.intersect1d(train_subjects,test1_subjects))
print(np.intersect1d(val_subjects,test0_subjects))
print(np.intersect1d(val_subjects,test1_subjects))
print(np.intersect1d(test0_subjects,test1_subjects))




# %%

data_set,_=give_oasis_data('train',task='sex',debug=False)
labels_train=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('val',task='sex',debug=False)
labels_val=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('test0',task='sex',debug=False)
labels_test0=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('test1',task='sex',debug=False)
labels_test1=np.array(data_set.label_list).flatten()
# %%
share_men=[labels_train.mean(), labels_val.mean(), 
        labels_test0.mean(),labels_test1.mean()]
plt.bar(["Train","Valid","Test0","Test1"],share_men,color=col_list)
plt.title("Share of male.")
plt.savefig("exploration/plots/oasis3_age_dist.png")
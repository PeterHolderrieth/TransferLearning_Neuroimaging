#%%
import numpy as np
import matplotlib.pyplot as plt 
#%matplotlib inline
from load_oasis3 import give_oasis_data

data_set,_=give_oasis_data('train',debug=False)
labels_train=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('val',debug=False)
labels_val=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('test0',debug=False)
labels_test0=np.array(data_set.label_list).flatten()

data_set,_=give_oasis_data('test1',debug=False)
labels_test1=np.array(data_set.label_list).flatten()



def give_array_summary(x):
    dict_={
    'mean': x.mean(),
    'median': np.median(x),
    'min': x.min(),
    'max': x.max(),
    'std': x.std(),
    }
    return(dict_)

fig,ax=plt.subplots(ncols=2,nrows=1)
#print(give_array_summary(labels))
n_bins=20
ax[0].hist(labels_train,bins=n_bins,alpha=0.6,density=True)
ax[0].hist(labels_val,bins=n_bins,alpha=0.6,density=True)
ax[1].hist(labels_test0,bins=n_bins,alpha=0.5,density=True)
ax[1].hist(labels_test1,bins=n_bins,alpha=0.6,density=True)

# %%

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet
import numpy as np
import data.oasis.load_oasis3 as load_oasis

def batch_fit_pca(data_loader,n_components):
    batch_size=data_loader.batch_size
    pca=IncrementalPCA(n_components=n_components,batch_size=batch_size)
    for batch_idx, (X,Y) in enumerate(data_loader):
        X=X.reshape(BATCH,np.prod(X.shape[1:]))
        pca.partial_fit(X)
    return(pca)

def batch_trans_pca(pca,data_loader):
    trans_list=[]
    age_list=[]
    for batch_idx, (X,Y) in enumerate(train_loader):
        X=X.reshape(BATCH,np.prod(X.shape[1:]))
        trans_list.append(pca.transform(X))
        age_list.append(Y.flatten())
    #Concatenate:
    data_trans=np.concatenate(trans_list,axis=0)
    label=np.concatenate(age_list,axis=0)
    return(data_trans,label)


BATCH=3
DEBUG=True
N_COMP=BATCH

#Load train loader:
_,train_loader=load_oasis.give_oasis_data('train',batch_size=BATCH,debug=DEBUG,shuffle=False)

#Get PCA of train data:
pca=batch_fit_pca(train_loader,N_COMP)

#Get transformed train data:
data_trans,label=batch_trans_pca(pca,train_loader)

#Fit ElasticNet:
eln=ElasticNet(l1_ratio=0.5)
eln.fit(data_trans,label)


#Reshape validation data set:
_,val_loader=load_oasis.give_oasis_data('val',batch_size=BATCH,debug=DEBUG,shuffle=False)
val_data_trans,val_label=batch_trans_pca(pca,val_loader)

#Predict:
Predic=eln.predict(val_data_trans)

#Get mean absolute error (mae):
mae=np.mean(np.abs(Predic-val_label))
mae_stupid=np.mean(np.abs(val_label-np.mean(label)))
mae_val=np.mean(np.abs(val_label-np.mean(val_label)))

#Print results:
print("Mean absolute: ",mae)
print("Mean absolute deviance of stupid guess: ", mae_stupid)
print("Mean absolute deviance of validation set: ", mae_val)





'''
def get_oasis_data_matrix(data_type='train',n_samples=10):
    scan_list=[]
    age_list=[]
    data_set,_=load_oasis.give_oasis_data(data_type,preprocessing='min')
    ind=np.random.permutation(np.arange(0,data_set._len))[:n_samples]
    for it in range(n_samples): 
        new_item=data_set.get_data(ind[it])
        x_new=new_item[0].flatten()
        scan_list.append(x_new)
        y_new=new_item[1]
        age_list.append(y_new)
    X=np.stack(scan_list)
    Y=np.array(age_list).flatten()
    X=scale(X)
    return(X,Y)
'''
import torch 
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression

import sys 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#A batch version of PCA:
def batch_fit_pca(data_loader,n_components):
    '''
    Inputs:
        data_loader - pytorch data loader giving data in batches
        n_components - int - number of principal components to use
    Output:
        pca - sklearn.decomposition.PCA - pca function fitted to data from data_loader
    '''
    batch_size=data_loader.batch_size
    
    pca=IncrementalPCA(n_components=n_components,batch_size=batch_size)

    for batch_idx, (X,Y) in enumerate(data_loader):
        X=X.reshape(batch_size,-1)
        pca.partial_fit(X)
    return(pca)

#A function transforming data from a data loader via PCA into a data matrix:
def batch_trans_pca(pca,data_loader):
    '''
    Inputs:
        pca - sklearn.decomposition.PCA - pca function 
        data_loader - pytorch data loader giving data in batches
    Output:
        np.array - shape (k,n_components) where n_components are given by the number of components from pca
    '''
    #Init empty lists:
    trans_list=[]
    age_list=[]

    batch_size=data_loader.batch_size

    for batch_idx, (X,Y) in enumerate(data_loader):

        #Reshape inputs:
        X=X.reshape(batch_size,-1)
        Y=Y.flatten()
        
        #Add transformed X and non-transformed Y:
        trans_list.append(pca.transform(X))
        age_list.append(Y)

    #Concatenate:
    data_trans=np.concatenate(trans_list,axis=0)
    label=np.concatenate(age_list,axis=0)

    return(data_trans,label)

def mat_corr_coeff(x,y):
    '''
    Input: 
        x - np.array - shape (n,p)
        y - np.array - shape (n)
    Output: 
       corrcoeff- np.array - shape (p) - coerrcoeff[i] gives Pearson correlation between
        x[:,i] and y
    '''    
    y_cent=y-y.mean()
    x_cent=x-x.mean(axis=0)[None,:]
    cov=np.dot(y_cent,x_cent)/x.shape[0]
    std_y=y.std()
    std_x=x.std(axis=0)
    corrcoeff=(cov/std_y)/std_x
    return(corrcoeff)

def give_most_correlative_features(x,y,n_features):
    corrcoeff=mat_corr_coeff(x,y)
    ind=corrcoeff.argsort()[-n_features:][::-1]
    return(ind)


def fit_elastic(train_loader,batch, ncomp, l1rat, reg, feat,method):
    #Set the number of PCA-components:
    ncomp=ncomp if ncomp>0 else train_loader.batch_size-1

    #Get PCA of train data:
    pca=batch_fit_pca(train_loader,ncomp)

    #Get transformed train data:
    data_trans,label=batch_trans_pca(pca,train_loader)

    if feat>0:
        #Give the indices of the most correlative indices:
        inds=give_most_correlative_features(data_trans, label,n_features=hp['feat'])
        #Select the features from the data:
        data_trans=data_trans[:,inds]
    else: 
        inds=None
    #Fit regression model:
    if method=='regression':
        reg_model=ElasticNet(alpha=reg,l1_ratio=l1rat)
    elif task=='logistic':
        reg_model=LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = l1rat,C=1/reg)
    else: 
        sys.exit("Unknown method. Either 'regression' or logistic.")

    reg_model.fit(data_trans,label)
    return(pca,reg_model,inds)

def test_elastic(val_loader,space,pca,reg_model,inds=None):
    #Get transformed validation data:
    val_data_trans,val_label=batch_trans_pca(pca,val_loader)

    if inds is not None:
        #Select correlative features:
        val_data_trans=val_data_trans[:,inds]

    #Predict on validation set:
    Predic=reg_model.predict(val_data_trans)

    if space=='continuous':
        #Get mean absolute error (mae):
        mae=np.mean(np.abs(Predic-val_label))

        #Get "stupid" baselines:
        mae_stupid=np.mean(np.abs(val_label-np.mean(label)))
        mae_val=np.mean(np.abs(val_label-np.mean(val_label)))
        #Print results:
        print("MAE of ElasticNet: ",mae)
        print("MAE on valid when train mean is predicted: ", mae_stupid)
        print("MAE on valid when valid mean is predicted: ", mae_val)
        return(mae_val)
        
    elif space=='binary':
        val_acc=((Predic>0.5)*val_label+(Predic<=0.5)*(1-val_label)).mean()
        print("Accuracy of regression: ", val_acc)
        return(val_acc)

    else:
        sys.exit("Unknown space. Either 'regression' or 'logistic'.")


def elastic_experiment(train_loader,hps,space):
    pca,reg_model,inds=fit_elastic(train_loader,**hps)
    result=test_elastic(val_loader,space,pca,reg_model,inds)
    return(result)
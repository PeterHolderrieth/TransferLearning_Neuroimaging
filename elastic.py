import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression

import argparse
import sys 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Own files:
import data.oasis.load_oasis3 as load_oasis


# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(
    BATCH=3,
    DEBUG='debug',
    N_COMP=None,
    L1RAT=0.5,
    REG=1.,
    FEAT=None,
    TASK='age')

ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="'debug' or 'full'.")
ap.add_argument("-batch", "--BATCH", type=int, required=True,help="Batch size.")
ap.add_argument("-ncomp", "--N_COMP", type=int, required=False,help="Number of principal components.")
ap.add_argument("-l1rat", "--L1RAT", type=float, required=False,help="Ratio of L1 loss (compared to L2).")
ap.add_argument("-reg", "--REG", type=float, required=False,help="Scaling of ElasticNet regularizer term.")
ap.add_argument("-task", "--TASK", type=float, required=False,help="Task: either 'age' or 'sex'.")
ap.add_argument("-feat", "--FEAT", type=int, required=False,help="Number of most correlative features to pick."+
                                                                    "If None, all features are picked.")

#Get arguments:
ARGS = vars(ap.parse_args())
print(ARGS)

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

#Set debug option:
if ARGS['DEBUG']=='debug':
    DEBUG=True
elif ARGS['DEBUG']=='full':
    DEBUG=False
else:
    sys.exit("Unknown debug option.")



#Load train loader:
_,train_loader=load_oasis.give_oasis_data('train',batch_size=ARGS['BATCH'],debug=DEBUG,shuffle=False,task=ARGS['TASK'])

#Set the number of PCA-components:
N_COMP=ARGS['N_COMP'] if ARGS['N_COMP'] is not None else train_loader.batch_size-1

#Get PCA of train data:
pca=batch_fit_pca(train_loader,N_COMP)

#Get transformed train data:
data_trans,label=batch_trans_pca(pca,train_loader)

if ARGS['FEAT'] is not None:
    #Give the indices of the most correlative indices:
    inds=give_most_correlative_features(data_trans, label,n_features=ARGS['FEAT'] )
    #Select the features from the data:
    data_trans=data_trans[:,inds]

#Fit regression model:
if ARGS['TASK']=='age':
    reg_model=ElasticNet(alpha=ARGS['REG'],l1_ratio=ARGS['L1RAT'])
elif ARGS['TASK']=='sex':
    reg_model=LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = ARGS['L1RAT'],C=1/ARGS['REG'])
else: 
    sys.exit("Unknown task. Either 'sex' or 'age'.")

reg_model.fit(data_trans,label)

#Get validation data set:
_,val_loader=load_oasis.give_oasis_data('val',batch_size=ARGS['BATCH'],debug=DEBUG,shuffle=False,task=ARGS['TASK'])

#Get transformed validation data:
val_data_trans,val_label=batch_trans_pca(pca,val_loader)

if ARGS['FEAT'] is not None:
    #Select correlative features:
    val_data_trans=val_data_trans[:,inds]

#Predict on validation set:
Predic=reg_model.predict(val_data_trans)

if ARGS['TASK']=='age':
    #Get mean absolute error (mae):
    mae=np.mean(np.abs(Predic-val_label))

    #Get "stupid" baselines:
    mae_stupid=np.mean(np.abs(val_label-np.mean(label)))
    mae_val=np.mean(np.abs(val_label-np.mean(val_label)))
    #Print results:
    print("MAE of ElasticNet: ",mae)
    print("MAE on valid when train mean is predicted: ", mae_stupid)
    print("MAE on valid when valid mean is predicted: ", mae_val)
else: 
    eval_func=dpl.give_my_loss_func({'type':'acc','thresh':0.5})
    val_acc=eval_func(torch.log(Predic),val_label)
    print("Accuracy of regression: ", val_acc)




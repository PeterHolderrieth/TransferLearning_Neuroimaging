import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet
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
    l1rat=0.5,
    reg=1.)

ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="'debug' or 'full'.")
ap.add_argument("-batch", "--BATCH", type=int, required=True,help="Batch size.")
ap.add_argument("-ncomp", "--N_COMP", type=int, required=False,help="Number of principal components.")
ap.add_argument("-l1rat", "--L1RAT", type=float, required=True,help="Ratio of L1 loss (compared to L2).")
ap.add_argument("-reg", "--REG", type=float, required=True,help="Scaling of ElasticNet regularizer term.")

#Get arguments:
ARGS = vars(ap.parse_args())
print(ARGS)

#A batch version of PCA:
def batch_fit_pca(data_loader,n_components):
    '''
    Inputs:
        data_loader - pytorch data loader giving data in batches
        n_components - int  - number of principal components to use
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


#Set debug option:
if ARGS['DEBUG']=='debug':
    DEBUG=True
elif ARGS['DEBUG']=='full':
    DEBUG=False
else:
    sys.exit("Unknown debug option.")

#Set the number of PCA-components:
N_COMP=ARGS['N_COMP'] if ARGS['N_COMP'] is not None else ARGS['BATCH']-1

#Load train loader:
_,train_loader=load_oasis.give_oasis_data('train',batch_size=ARGS['BATCH'],debug=DEBUG,shuffle=False)

#Get PCA of train data:
pca=batch_fit_pca(train_loader,N_COMP)

#Get transformed train data:
data_trans,label=batch_trans_pca(pca,train_loader)

#Fit ElasticNet:
eln=ElasticNet(alpha=ARGS['REG'],l1_ratio=ARGS['L1RAT'])
eln.fit(data_trans,label)

#Get validation data set:
_,val_loader=load_oasis.give_oasis_data('val',batch_size=ARGS['BATCH'],debug=DEBUG,shuffle=False)

#Get transformed validation data:
val_data_trans,val_label=batch_trans_pca(pca,val_loader)

#Predict on validation set:
Predic=eln.predict(val_data_trans)

#Get mean absolute error (mae):
mae=(Predic-val_label).abs().mean()

#Get "stupid" baselines:
mae_stupid=(val_label-np.mean(label)).abs().mean()
mae_val=(val_label-np.mean(val_label)).abs().mean()

#Print results:
print("MAE of ElasticNet: ",mae)
print("MAE on valid when train mean is predicted: ", mae_stupid)
print("MAE on valid when valid mean is predicted: ", mae_val)
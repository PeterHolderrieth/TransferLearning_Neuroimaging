from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet
import numpy as np
import data.oasis.load_oasis3 as load_oasis
import argparse
import sys 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(
    BATCH=3,
    DEBUG='debug',
    N_COMP=None,
    l1rat=0.5,
    reg=1.)

#Debugging? Then use small data set:
ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="'debug' or 'full'.")
ap.add_argument("-batch", "--BATCH", type=int, required=True,help="Batch size.")
ap.add_argument("-ncomp", "--N_COMP", type=int, required=False,help="Number of principal components.")
ap.add_argument("-l1rat", "--L1RAT", type=float, required=True,help="Ratio of L1 loss (compared to L2).")
ap.add_argument("-reg", "--REG", type=float, required=True,help="Scaling of elastic regularizer.")


#Arguments for tracking:
ARGS = vars(ap.parse_args())
print(ARGS)

def batch_fit_pca(data_loader,n_components):
    batch_size=data_loader.batch_size
    pca=IncrementalPCA(n_components=n_components,batch_size=batch_size)
    for batch_idx, (X,Y) in enumerate(data_loader):
        X=X.reshape(ARGS['BATCH'],np.prod(X.shape[1:]))
        pca.partial_fit(X)
    return(pca)

def batch_trans_pca(pca,data_loader):
    trans_list=[]
    age_list=[]
    for batch_idx, (X,Y) in enumerate(train_loader):
        X=X.reshape(ARGS['BATCH'],np.prod(X.shape[1:]))
        trans_list.append(pca.transform(X))
        age_list.append(Y.flatten())
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


#Reshape validation data set:
_,val_loader=load_oasis.give_oasis_data('val',batch_size=ARGS['BATCH'],debug=DEBUG,shuffle=False)
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
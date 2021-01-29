from sklearn.decomposition import PCA  
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNet
import numpy as np
import data.oasis.load_oasis3 as load_oasis

def get_oasis_data(data_type='train',n_samples=10):
    scan_list=[]
    age_list=[]
    data_set,_=load_oasis.give_oasis_data(data_type,preprocessing='min')
    for it in range(n_samples): 
        new_item=data_set.get_data(it)
        x_new=new_item[0].flatten()
        scan_list.append(x_new)
        y_new=new_item[1]
        age_list.append(y_new)
    X=np.stack(scan_list)
    Y=np.array(age_list).flatten()
    X=scale(X)
    return(X,Y)

X_tr,Y_tr=get_oasis_data('train',10)
pca=PCA(n_components=2)
X_tr_trans=pca.fit_transform(X_tr)
eln=ElasticNet()
eln.fit(X_tr_trans,Y_tr)
print(eln.coef_)

X_val,Y_val=get_oasis_data('val',10)
X_val_trans=pca.transform(X_val)
Predic=eln.predict(X_val_trans)
mse=np.mean(np.())
print("Mean squared error: ", mse)


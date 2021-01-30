import numpy as np

x=np.random.randn(20,10)
y=np.random.randn(20)

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


print(give_most_correlative_features(x,y,4))
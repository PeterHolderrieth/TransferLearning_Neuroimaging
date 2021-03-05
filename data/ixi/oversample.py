import pandas as pd 
import numpy as np 

#Set seed:
np.random.seed(301200)

#Files:
folder='/gpfs3/well/win-fmrib-analysis/users/lhw539/ixi/'
filenames=['ixi_train','ixi_val','ixi_test']

for filename in filenames:
    filepath=folder+filename
    df=pd.read_csv(filepath+'.csv')
    
    df_female=df[df["Sex"]==0]
    df_male=df[df["Sex"]==1]

    n_female=df_female.shape[0]
    n_male=df_male.shape[0]
    
    print("Original data set: number of female: ", n_female)
    print("Original data set: number of male: ", n_male)

    factor=n_female/n_male
    factor_int=np.floor(factor).astype(int)
    random_share=factor-factor_int
    n_rand_choice=np.round(random_share*n_male).astype(int)
    rand_subset=np.random.permutation(n_male)[:n_rand_choice].astype(int)
    df_balanced=pd.concat([df_male.iloc[rand_subset]]+[df_male for it in range(factor_int)]+[df_female])
    print("Balanced values: ", np.unique(df_balanced["Sex"],return_counts=True))
    new_file_path=filepath+'_balanced_'+'sex'+'.csv'
    df_balanced.to_csv(new_file_path)
    print("Saved to: ", new_file_path)
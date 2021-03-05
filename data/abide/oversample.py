import pandas as pd 
import numpy as np 

#Set seed:
np.random.seed(301200)


#Files:
folder='/gpfs3/well/win-fmrib-analysis/users/lhw539/abide/'
filenames=['info/abide_train','info/abide_val','info/abide_test']


for filename in filenames:
    filepath=folder+filename
    df=pd.read_csv(filepath+'.csv')
    
    df_female=df[df["Sex"]==0]
    df_male=df[df["Sex"]==1]

    n_females=df_female.shape[0]
    n_males=df_male.shape[0]
    
    print("Original data set: number of female: ", n_females)
    print("Original data set: number of male: ", n_males)

    factor=n_males/n_females
    factor_int=np.floor(factor).astype(int)
    random_share=factor-factor_int
    n_rand_choice=np.round(random_share*n_females).astype(int)
    rand_subset=np.random.permutation(n_females)[:n_rand_choice].astype(int)
    df_balanced=pd.concat([df_female.iloc[rand_subset]]+[df_female for it in range(factor_int)]+[df_male])
    print("Balanced values: ", np.unique(df_balanced["Sex"],return_counts=True))
    new_file_path=filepath+'_balanced_'+'sex'+'.csv'
    df_balanced.to_csv(new_file_path)
    print("Saved to: ", new_file_path)


for filename in filenames:
    filepath=folder+filename
    df=pd.read_csv(filepath+'.csv')
    
    df_disease=df[df["IsNC"]==0]
    df_isnc=df[df["IsNC"]==1]

    n_disease=df_disease.shape[0]
    n_isnc=df_isnc.shape[0]

    print("Original data set: number of people with disease: ", n_disease)
    print("Original data set: number of people without disease: ", n_isnc)

    factor=n_isnc/n_disease
    factor_int=np.floor(factor).astype(int)
    random_share=factor-factor_int
    n_rand_choice=np.round(random_share*n_disease).astype(int)
    rand_subset=np.random.permutation(n_disease)[:n_rand_choice].astype(int)
    df_balanced=pd.concat([df_disease.iloc[rand_subset]]+[df_disease for it in range(factor_int)]+[df_isnc])
    print("Values in balanced dataset: ", np.unique(df_balanced["IsNC"],return_counts=True)[1])
    new_file_path=filepath+'_balanced_'+'isnc'+'.csv'
    df_balanced.to_csv(new_file_path)
    print("Saved to: ", new_file_path)

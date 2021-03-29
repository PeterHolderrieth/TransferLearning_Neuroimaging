import pandas as pd 
import numpy as np 

#Set seed:
np.random.seed(3012)


#Files: 
folder='/well/win-fmrib-analysis/users/lhw539/oasis3/oasis3_info/'
filenames=['session_train','session_val','session_test0','session_test1']


for filename in filenames:
    filepath=folder+filename
    df=pd.read_csv(filepath+'.csv')
    
    #Get subject info:
    subject_df=pd.read_csv(folder+'subject_info.csv')
    
    #Extract subject and sex info. Set subject as index:
    subject_sex=subject_df[["Subject","Sex"]].set_index("Subject")
    
    #Extract labels:
    label_list=[sex_.item()  for sex_ in subject_sex.loc[df.Subject.values].values]
    df_female=df[[label==0 for label in label_list]]
    df_male=df[[label==1 for label in label_list]]

    n_females=df_female.shape[0]
    n_males=df_male.shape[0]
    
    print("Original data set: number of female: ", n_females)
    print("Original data set: number of male: ", n_males)

    if n_males>n_females:
        factor=n_males/n_females
        factor_int=np.floor(factor).astype(int)
        random_share=factor-factor_int
        n_rand_choice=np.round(random_share*n_females).astype(int)
        rand_subset=np.random.permutation(n_females)[:n_rand_choice].astype(int)
        df_balanced=pd.concat([df_female.iloc[rand_subset]]+[df_female for it in range(factor_int)]+[df_male])
    else:
        factor=n_females/n_males
        factor_int=np.floor(factor).astype(int)
        random_share=factor-factor_int
        n_rand_choice=np.round(random_share*n_males).astype(int)
        rand_subset=np.random.permutation(n_males)[:n_rand_choice].astype(int)
        df_balanced=pd.concat([df_male.iloc[rand_subset]]+[df_male for it in range(factor_int)]+[df_female])

    #Extract labels:
    label_arr=np.array([sex_.item()  for sex_ in subject_sex.loc[df_balanced.Subject.values].values])
    print("Balanced values: ", np.unique(label_arr,return_counts=True))

    print("Age distribution: ")
    print("-----------------")
    print("Minimum: ", df_balanced["Age"].min())
    print("Maximum: ",df_balanced["Age"].max())
    print("Mean: ",df_balanced["Age"].mean())
    print("Std: ",df_balanced["Age"].std())
    print("-----------------")

    new_file_path=filepath+'_balanced_'+'sex'+'.csv'
    df_balanced.to_csv(new_file_path)
    print("Saved to: ", new_file_path)

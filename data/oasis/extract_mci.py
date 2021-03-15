
import pandas as pd 
import numpy as np
DIR = '/gpfs3/well/win-fmrib-analysis/users/lhw539/oasis3/oasis3_info/'
fp_ = DIR+'session_info_clinical.csv'    
df_session = pd.read_csv(fp_)

df_mci=df_session.loc[(df_session["cdr"]>0)&(df_session["cdr"]<2)]
subject_ids,counts=np.unique(df_mci["Subject"],return_counts=True)
print("Number of subjects with two or more sessions:", counts[counts>=].shape[0])


#print(df_session[["Subject","Age","ageAtEntry","cdr"]].head())

#print("Shape of data frame: ", df_session.shape)
#print("Variables: ", list(df_session.columns))

'''
#Get all possible CDR scores:
cdr_val=np.unique(df_session["cdr"].to_numpy()).astype(float)
print("Possible CDR scores: ", cdr_val)

df_session["cdr_class"]=np.nan

#Count the number of subjects in each category:
for it in range(cdr_val.size):
    cdr=cdr_val[it]
    n_subjects=df_session.loc[df_session["cdr"]==cdr].shape[0]
    print("%4d sessions with CDR score of %.1f"%(n_subjects,cdr))
'''
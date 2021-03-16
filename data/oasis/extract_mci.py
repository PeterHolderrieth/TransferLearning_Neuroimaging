
import pandas as pd 
import numpy as np

#Load clinical info of data:
DIR = '/gpfs3/well/win-fmrib-analysis/users/lhw539/oasis3/oasis3_info/'
fp_clinical = DIR+'session_info_clinical.csv'    
df_clinical = pd.read_csv(fp_clinical)

print(list(df_clinical.columns))

#Exctract subjects with cdr score=0.5,1 
df_mci=df_clinical[(df_clinical["cdr"]>0.1)&(df_clinical["cdr"]<1.9)]
subject_ids,counts=np.unique(df_mci["Subject"],return_counts=True)
print("Number of subjects with two or more sessions and CDR>0:", counts[counts>1].shape[0])

#Exctract all subjects with more than one MRI session ('non-unique subject'):
non_unique_subjects=[]
for idx in range(len(counts)):
    if counts[idx]>1:
        non_unique_subjects.append(subject_ids[idx])

#Set subject as index:
df_mci=df_mci.set_index("Subject")

valid_follow_up_clinical_id=[]
prog_mci=[]
age_follow_up=[]
age_now=[]

for subject in non_unique_subjects:
    
    df_subject=df_mci.loc[subject]
    n_sessions=df_subject.shape[0]
    
    for it in range(n_sessions-1):

        #Extract time difference (in years) between sessions:
        time_difference=df_subject["Age"][it+1]-df_subject["Age"][it]

        #A follow-up is only valid if approximately a year has passed
        if time_difference>0.9:
            valid_follow_up_clinical_id.append(df_subject["ADRC_ADRCCLINICALDATA ID"][it])
            age_follow_up.append(df_subject["Age"][it+1])
            age_now.append(df_subject["Age"][it])
            #If cdr score has increased, then it is progressive MCI, otherwise it is stable:
            if df_subject["cdr"][it+1]-df_subject["cdr"][it]>0:
                prog_mci.append(1)
            else: 
                prog_mci.append(0)


fp_session = DIR+'session_info.csv'    
df_session = pd.read_csv(fp_session)
df_session=df_session.set_index("ADRC_ADRCCLINICALDATA ID")

df_session=df_session.loc[valid_follow_up_clinical_id]

df_session["AgeAtSession"]=age_now
df_session["AgeFollowUp"]=age_follow_up
df_session["ProgMCI"]=prog_mci


df_session.reset_index(level=0,inplace=True)
print("Columns df sessions: ", list(df_session.columns))

n_progressive_mci=df_session["ProgMCI"].sum()
print("Number of subjects with progressive MCI: ",n_progressive_mci )
print("Share of subjects with progressive MCI: ", n_progressive_mci/df_session.shape[0])

fp_mci = DIR+'mci_info.csv'  
df_session.to_csv(fp_mci)

subjects=np.unique(df_session["Subject"])
n_subjects=len(subjects)
subjects=subjects[np.random.permutation(n_subjects)]
share_train=0.6
share_valid=0.2

n_train=int(share_train*n_subjects)
n_valid=int(share_valid*n_subjects)
n_test=n_subjects-n_valid-n_train

subjects_train=subjects[:n_train].tolist()
subjects_valid=subjects[n_train:(n_valid+n_train)].tolist()
subjects_test=subjects[(n_valid+n_train):].tolist()

print("Number of train subjects: ", n_train)
print("Number of valid subjects: ", n_valid)
print("Number of test subjects: ", n_test)

print(subjects_train)
df_session.set_index("Subject",inplace=True)

df_train=df_session.loc[subjects_train].reset_index(level=0)
df_valid=df_session.loc[subjects_valid].reset_index(level=0)
df_test=df_session.loc[subjects_test].reset_index(level=0)

print("Number of train sessions: ", df_train.shape[0])
print("Number of valid sessions: ", df_valid.shape[0])
print("Number of test sessions: ", df_test.shape[0])


'''
print("Share of male in train: ", df_train["sex"])
print("Share of male in valid: ", df_valid.shape[0])
print("Share of male in test: ", df_test.shape[0])
'''

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
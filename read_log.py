import numpy as np 
import pandas as pd

def read_log(filepath):
    with open(filepath,"r") as fi:
        rows_list = []
        for ln in fi:
            if ln.startswith("|epoch:"):
                rows_list.append(ln)
        #column_names=["epoch","lr","train kld ravg","train mae ravg","val kldravg","val mae ravg"]
        column_names=["epoch","lr","train ent ravg","train acc ravg","val entravg","val acc ravg"]
        df=pd.DataFrame(columns=column_names)
        for idx, row in enumerate(rows_list):
            for search_string in column_names:
                start_pos=row.find(search_string)
                offset=start_pos+len(search_string)+1
                end_pos=row.find("|",offset)
                value=float(row[offset:end_pos])
                print(search_string)
                print(value)
                df.loc[idx,search_string]=value
        return df

#filepath='experiments/abide/sex/sex_pretrained/run_1_final_abide_sex_init_with_scaling/run_1_final_abide_sex_init_with_scaling20210403_1232'
#filepath='experiments/abide/sex/sex_pretrained/run_1_final_ft_full_sex_abide_sex_pretrained/run_1_final_ft_full_sex_abide_sex_pretrained20210326_1108'
#filepath='experiments/oasis/sex/sex_pretrained/final_1_oasis_sex_sex_pretrained/final_1_oasis_sex_sex_pretrained20210319_1147'
#filepath='experiments/oasis/sex/scratch/final_1_scratch_oasis_sex/final_1_scratch_oasis_sex20210319_1212'
#filepath='experiments/abide/sex/sex_pretrained/run_3_final_ft_full_sex_abide_sex_pretrained/run_3_final_ft_full_sex_abide_sex_pretrained20210326_1108'
#filepath='experiments/ixi/sex/scratch/run_1_final_ixi_sex_scratch/run_1_final_ixi_sex_scratch20210324_1502'
filepath='experiments/ixi/sex/sex_pretrained/run_1_final_ft_full_sex_sex_pretrained_ixi/run_1_final_ft_full_sex_sex_pretrained_ixi20210329_1733'
#filepath='experiments/oasis/age/age_pretrained/run_1_final_oasis_age_age_pretrained_ft_step/run_1_final_oasis_age_age_pretrained_ft_step20210326_1022' 

df=read_log(filepath+'.log')
df.to_csv(filepath+'.csv')
import pandas as pd 
import sys 
import numpy as np
import argparse

PATH='hyperparameters.csv'
SPACES=['disc','cont']
TYPES=['string','float','int']


def add_hyperparameter(name,type_,space,val_list=None,path=PATH):
    #Control whether type and space are valid:
    if type_ not in SPACES or space not in SPACES:
        sys.exit("Invalid type or space.")

    df = pd.read_csv(path,index_col=None)
    nrows,ncols=df.shape
    if name in list(df.Name):
        sys.exit("Hyperparameter '"+name+"' already exists.")
    df.loc[nrows,"Name"]=name 
    df.loc[nrows,"Type"]=type_
    df.loc[nrows,"Space"]=space  
    if val_list is not None:
        n_vals=len(val_list)
        for it in range(n_vals):
            df.loc[nrows,"V"+str(it)]=val_list[it]
    df.to_csv(path,index=False)

def remove_hyperparameter(name,path=PATH):
    df=pd.read_csv(path,index_col=None)
    df.drop(df.loc[df.Name==name].index, inplace=True)
    df.reset_index()
    df.to_csv(path,index=False)

def add_possible_value(name,value,path=PATH):
    df=pd.read_csv(path,index_col=None)
    #If value already exists, ignore:
    if (df.loc[df.Name==name].iloc[:,3:]==value).values.any():
        return
    #Get the minimum index which is NaN:
    vec=df.loc[df.Name==name].isnull().values[0]
    nan_list= [it for it, val in enumerate(vec) if val]

    #Fill this index:
    if len(nan_list)>0:
        min_ind_nan=nan_list[0]
        col="V"+str(min_ind_nan-3)
    else: 
        col="V"+str(df.shape[1]-3)

    df.loc[df.Name==name,col]=value
    df.to_csv(path,index=False)
  



# Construct the argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-act", "--ACTION", type=str, required=True,help="Either 'adp'(add parameter),\
                                                                'adv' (add value),\
                                                                'rmp' (remove parameter)\
                                                                or 'p' (print data frame).\
                                                                or 'ph' (print head of data frame).") 
ap.add_argument("-name", "--NAME", type=str, required=False,help="Name of hyperparameter")
ap.add_argument("-type", "--TYPE", type=str, required=False,help="Type of hyperparameter")
ap.add_argument("-space", "--SPACE", type=str, required=False,help="Space of hyperparameter")
ap.add_argument("-val", "--VALUE", type=str, required=False,help="Value.")

#Arguments for tracking:
ARGS = vars(ap.parse_args())

if ARGS['ACTION']=='adp':
    add_hyperparameter(name,type_,space,val_list=[ARGS['VALUE']])
elif ARGS['ACTION']=='adv':
    add_possible_value(name,value=ARGS['VALUE'])
elif ARGS['ACTION']=='p':
    df=pd.read_csv(PATH,index_col=None)
    print(df)
elif ARGS['ACTION']=='ph':
    df=pd.read_csv(PATH,index_col=None)
    print(df.head())
else: 
    sys.exit("Unknown action.")

  
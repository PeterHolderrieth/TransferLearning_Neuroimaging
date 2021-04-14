import torch 
from data.oasis.load_oasis3 import give_oasis_data
from data.uk_biobank.load_ukb import give_ukb_data
from sfcn.sfcn_load import give_pretrained_sfcn
import numpy as np
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
import pandas as pd 
import argparse 
from data.load_dataset import give_dataset


label='age'
source='oasis'
for label in ['sex','age']:
    for source in ['oasis','abide','ixi']:
        train_dataset,_=give_dataset(source,'train', 
                                                batch_size=1,
                                                num_workers=4,
                                                shuffle=True,
                                                debug=False,
                                                preprocessing='min',
                                                task=label,
                                                share=1.0,
                                                balance=True)

        test_dataset,_=give_dataset(source,'test', 
                                                batch_size=1,
                                                num_workers=4,
                                                shuffle=True,
                                                debug=False,
                                                preprocessing='min',
                                                task=label,
                                                share=1.0,
                                                balance=True)
            

        train_labels=np.array([label for label in train_dataset.label_list])
        test_labels=np.array([label for label in test_dataset.label_list])
        
        train_mean=train_labels.mean()

        if label=='age':

            error=np.abs(test_labels-train_mean).mean()

            print("Prediction error for age and data %s: %.4f "%(source,error))

        elif label=='sex':
            more_male=bool(train_mean>0.5)

            if more_male:
                error=train_mean
            else: 
                error=1-train_mean

            print("Prediction error for sex and data %s: %.4f "%(source,error))


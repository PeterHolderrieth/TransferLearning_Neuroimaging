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

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(RUN="0",
                DATA="ukb",
                METHOD='pca',
                LOAD='age',
                LABEL='age',
                N_SAMPLES=300)

ap.add_argument("-deb", "--DEBUG", type=str, required=True,help="'debug' or 'full'.")
ap.add_argument("-dat", "--DATA", type=str, required=False,help="Data: either 'oasis' or 'ukb'.")
ap.add_argument("-met", "--METHOD", type=str, required=False,help="Method to use: possible is 'pca','tsne','isomap', or 'all'.") 
ap.add_argument("-run", "--RUN", type=str, required=False,help="Runs to use. Either '0',...,'4' or 'all'")
ap.add_argument("-load", "--LOAD", type=str, required=False,help="Either 'age' or 'sex'. Reload model pre-trained on this task.")
ap.add_argument("-lab", "--LABEL", type=str, required=False,help="Either 'age', 'sex' or all. Label to include in dataset.")
ap.add_argument("-nsam", "--N_SAMPLES", type=int, required=False,help="Number of samples to include.")


#Get arguments:
ARGS = vars(ap.parse_args())

#Look up freesurfer.
allowed_methods=['tsne']#,'pca','isomap']
allowed_labels=['sex']
allowed_loads=['sex']
#allowed_runs=['0','1','2','3','4']
#allowed_runs=["abide_3"]#,"oasis_1"]
allowed_runs=["oasis_1"]


#Get the methods we use:
if ARGS['METHOD']!='all' and (ARGS['METHOD'] not in allowed_methods):
    sys.exit("Unknown method.")
elif ARGS['METHOD']!='all':
    methods=[ARGS['METHOD']]
else: 
    methods=allowed_methods

#Get the labels we use:
if ARGS['LABEL']!='all' and (ARGS['LABEL'] not in allowed_labels):
    sys.exit("Unknown LABEL.")
elif ARGS['LABEL']!='all':
    labels=[ARGS['LABEL']]
else: 
    labels=allowed_labels

#Get the tasks the models were pre-trained on:
if ARGS['LOAD']!='all' and (ARGS['LOAD'] not in allowed_loads):
    sys.exit("Unknown LOAD.")
elif ARGS['LOAD']!='all':
    pretrained_tasks=[ARGS['LOAD']]
else: 
    pretrained_tasks=allowed_loads

#Get the runs we use:
if ARGS['RUN']!='all' and (ARGS['RUN'] not in allowed_runs):
    sys.exit("Unknown RUN.")
elif ARGS['RUN']!='all':
    runs=[ARGS['RUN']]
else: 
    runs=allowed_runs


for label in labels:
    data_type='test' 
            
    train_dataset,_=give_dataset(ARGS['DATA'],data_type, 
                                            batch_size=1,
                                            num_workers=4,
                                            shuffle=True,
                                            debug=False,
                                            preprocessing='min',
                                            task=label,
                                            share=1.0,
                                            balance=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs: ", torch.cuda.device_count())

    for method in methods:
        for run in runs:
            for pretrained_task in pretrained_tasks:

                model=give_pretrained_sfcn(run, pretrained_task)
                model.eval()
                model.module.train_nothing() 
                model=model.to(device)

                if ARGS['DEBUG']=='debug':
                    n_samples=10
                elif ARGS['DEBUG']=='full':
                    n_samples=min(train_dataset._len, ARGS['N_SAMPLES'])
                else: 
                    sys.exit("Unknown debug flag. Either 'debug' or 'full'.")

                #Get random subset of indices of the dataset:
                size_dataset=train_dataset._len
                inds=torch.randperm(size_dataset)[:n_samples].tolist()
                n_final_channels=model.module.channel_number[-1]
                feat_matrix=np.zeros(shape=(n_samples,n_final_channels))

                for it in range(n_samples):
                    idx=inds[it]
                    print("Iteration: %5d | Element: %5d"%(it,idx))
                    data,_=train_dataset.get_data(idx)
                    data=torch.tensor(data[np.newaxis,...],dtype=torch.float32,requires_grad=False)
                    data=data.to(device)
                    output=model.module.feature_extractor(data)
                    output=model.module.classifier.average_pool(output)
                    feat_matrix[it]=output.flatten().detach().cpu().numpy()
                
                #fp=[train_dataset.file_list[ind] for ind in inds]

                if method=='tsne':
                    tsne=TSNE(n_components=2, init='pca')
                    print("Compute tSNE.")
                    trans_feat_matrix=tsne.fit_transform(feat_matrix)
                    print("Finished computing tSNE.")
                
                elif method=='isomap':
                    isomap=Isomap()
                    print("Compute isomap.")
                    trans_feat_matrix=isomap.fit_transform(feat_matrix)
                    print("Finished computing isomap.")
                
                elif method=='pca':
                    pca=PCA(n_components=2)
                    print("Compute pca.")
                    trans_feat_matrix=pca.fit_transform(feat_matrix)
                    print("Finished computing pca.")

                df=pd.DataFrame(trans_feat_matrix,columns=['COMP1','COMP2'])
                df['fp']=[train_dataset.file_list[idx] for idx in inds]
                df['label']=[float(train_dataset.label_list[idx][0]) for idx in inds]

                folder="/well/win-fmrib-analysis/users/lhw539/visualization/"
                filename=str(ARGS['DATA'])+"_run_"+str(run)+'_'+method+'_label'+label+'_pretrained_on_'+str(pretrained_task)+'_'+str(n_samples)+'_samples.csv'
                if ARGS['DEBUG']=='full':
                    print("Saving csv file to: ", folder+filename)
                    df.to_csv(folder+filename)

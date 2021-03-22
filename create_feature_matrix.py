import torch 
from data.oasis.load_oasis3 import give_oasis_data
from data.uk_biobank.load_ukb import give_ukb_data
from sfcn.sfcn_load import give_pretrained_sfcn
import numpy as np
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
import pandas as pd 

#Look up freesurfer.

debug=False
methods=['pca','tsne','isomap']

# train_dataset,_=give_oasis_data('train', batch_size=5,
#                                         num_workers=4,
#                                         shuffle=False,
#                                         debug=debug,
#                                         preprocessing='min',
#                                         task='age',
#                                         share=1.)

train_dataset,_=give_ukb_data('train', batch_size=5,
                                        num_workers=4,
                                        shuffle=False,
                                        debug=debug,
                                        preprocessing='min',
                                        task='age',
                                        share=300/40000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of GPUs: ", torch.cuda.device_count())

for method in methods:
    for run in range(4):
        model=give_pretrained_sfcn(str(run), "sex")
        model.eval()
        model.module.train_nothing() 
        model=model.to(device)

        n_samples=train_dataset._len
        n_final_channels=model.module.channel_number[-1]
        feat_matrix=np.zeros(shape=(n_samples,n_final_channels))

        for idx in range(n_samples):
            print("Element: ", idx)
            data,_=train_dataset.get_data(idx)
            data=torch.tensor(data[np.newaxis,...],dtype=torch.float32,requires_grad=False)
            data=data.to(device)
            output=model.module.feature_extractor(data)
            output=model.module.classifier.average_pool(output)
            feat_matrix[idx]=output.flatten().detach().cpu().numpy()

        fp=train_dataset.file_list

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

        df=pd.DataFrame(trans_feat_matrix,columns=['tSNE1','tSNE2'])
        df['fp']=fp
        df['label']=[float(label[0]) for label in train_dataset.label_list]

        folder="visualization/data/"
        filename="ukb_run_"+str(run)+'_'+method+'.csv'
        df.to_csv(folder+filename)

from data.load_dataset import give_dataset
import numpy as np 
for data_type in ['train','val','test']:
    dataset,_=give_dataset('oasis',data_type, batch_size=1,
                                                num_workers=4,
                                                shuffle=True,
                                                debug=False,
                                                preprocessing='full',
                                                task='age',
                                                share=1.0,
                                                balance=False)
    x=np.array([label[0] for label in dataset.label_list])
    print(x.mean())    
    print(x.std())
    print(x.max())
    print(x.min())
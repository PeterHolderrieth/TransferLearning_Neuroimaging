from data.load_dataset import give_dataset
import numpy as np 

data_source='oasis'
data_type='train'
label='age'
for label in ['age','sex']:
    for data_type in ['train','val','test']:
        dataset,_=give_dataset(data_source,data_type, 
                                                batch_size=1,
                                                num_workers=4,
                                                shuffle=True,
                                                debug=False,
                                                preprocessing='min',
                                                task=label,
                                                share=1.0,
                                                balance=False)
        y=np.array(dataset.label_list)
        print()
        print("Label: ", label)
        print("Set: ", data_type)
        print("Size: ", len(y))
        print("Min: ", y.min())
        print("Max: ", y.max())
        print("Mean: ", y.mean())
        print("Sum: ", y.sum())
        print("Length minus sum: ", len(y)-y.sum())
        print("Std: ", y.std())
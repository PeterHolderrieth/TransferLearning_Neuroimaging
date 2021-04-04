from data.load_dataset import give_dataset

data_source='oasis'
data_type='train'
label='age'
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

print("Min: ", y.min())
print("Max: ", y.max())
print("Mean: ", y.mean())
print("Std: ", y.std())
model=give_pretrained_sfcn("0", "age")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of GPUs: ", torch.cuda.device_count())
model=model.to(device)

_,train_loader=give_oasis_data('train', batch_size=1,
                                        num_workers=4,
                                        shuffle=True,
                                        debug='debug',
                                        preprocessing='min',
                                        task='age',
                                        share=1.)

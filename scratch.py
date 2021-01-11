import os
import os.path as osp

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys

sys.path.append('/users/win-biobank/jdo465/deep_medicine/')

import dm_model.dm_models as dmm
import dm_model.dm_utils as dmu
import dm_model.dm_loss as dml
import nibabel
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim

EXP_DIR = '/well/win-biobank/users/jdo465/age_sex_prediction/oasis3/run_20191206/'


def get_df_by_list(df=None, key='Subject', ref_list=None, exclusion=False):
    """
    if exclusion is True:
        return df.loc[[(v not in ref_list) for v in df[key]]]
    elif exclusion is False:
        return df.loc[[(v in ref_list) for v in df[key]]]
    """
    if exclusion is True:
        return df.loc[[(v not in ref_list) for v in df[key]]]
    elif exclusion is False:
        return df.loc[[(v in ref_list) for v in df[key]]]


# Training

# Code for dataset
class Dataset(torch.utils.data.Dataset):
    """
    config['TrainingInfo']['Preprocessing']:
    ( 'noise','mirror')
    """

    def __init__(self,
                 file_list,
                 label_list,
                 preprocessing=None):
        self.file_list = file_list
        self.label_list = label_list
        self._len = len(file_list)
        self.preprocessing = preprocessing

    def get_data(self, idx):
        label = self.label_list[idx]
        fp_ = self.file_list[idx]
        x = nibabel.load(fp_).get_fdata()
        if self.preprocessing is not None:
            for func_ in self.preprocessing:
                x = func_(x)
        data = np.reshape(x, (1,) + x.shape)
        return data, label

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        data, label = self.get_data(idx)
        data = torch.tensor(data, dtype=torch.float32, requires_grad=False)
        return data, label


def construct_preprocessing(kwd):
    """
    kwd: dictionary
     {'method': 'pixel_shift',
      'x_shift': 2,
      'y_shift': 2,
      'z_shift': 2}
     {'method': 'mirror',
      'probability': 0.5}
     {'method': 'average'}
     {'method': 'crop',
     'nx': 160,
     'ny': 192,
     'nz': 160}
    """
    if kwd['method'] == 'pixel_shift':

        def pixel_shift(data, x0=kwd['x_shift'], y0=kwd['y_shift'], z0=kwd['z_shift']):
            x_shift = int(torch.randint(low=-x0, high=x0 + 1, size=(1, 1)).numpy()[0, 0])
            y_shift = int(torch.randint(low=-y0, high=y0 + 1, size=(1, 1)).numpy()[0, 0])
            z_shift = int(torch.randint(low=-z0, high=z0 + 1, size=(1, 1)).numpy()[0, 0])
            data = np.roll(data, x_shift, axis=0)
            data = np.roll(data, y_shift, axis=1)
            data = np.roll(data, z_shift, axis=2)
            return data

        return pixel_shift

    elif kwd['method'] == 'mirror':

        def mirror(data, p0=kwd['probability']):
            p = torch.rand([1, 1]).numpy()[0, 0]
            if p < p0:
                data = np.flip(data, 0)
            return data

        return mirror

    elif kwd['method'] == 'average':

        def average(data):
            data = data / np.mean(data) - 1
            return data

        return average

    elif kwd['method'] == 'crop':

        def crop(data, nx=kwd['nx'], ny=kwd['ny'], nz=kwd['nz']):
            nx0, ny0, nz0 = data.shape
            x_start = np.ceil((nx0 - nx) / 2).astype(np.int)
            x_end = - np.floor((nx0 - nx) / 2).astype(np.int)
            y_start = np.ceil((ny0 - ny) / 2).astype(np.int)
            y_end = - np.floor((ny0 - ny) / 2).astype(np.int)
            z_start = np.ceil((nz0 - nz) / 2).astype(np.int)
            z_end = - np.floor((nz0 - nz) / 2).astype(np.int)
            data = data[x_start:x_end, y_start:y_end, z_start:z_end]
            return data

        return crop

    else:
        raise Exception('method \'' + kwd['method'] + '\' not support')


def go_one_epoch(state, model, loss_func, unpack_func, device, data_loader, metric_func=None, optimizer=None,
                 record_output=False):
    if state == 'train':
        model.train()
    elif state == 'test':
        model.eval()
    else:
        raise (f'train or test? Received {state}')
    # Init logging
    output_all = list()
    label_all = list()
    n_total = 0
    loss_total = 0
    metric_total = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        n_batch = data.shape[0]
        if state == 'train':
            target = unpack_func(label, device)
            optimizer.zero_grad()
            output = model(data)
            loss, loss_list = loss_func(output, target, device)
            loss.backward()
            optimizer.step()
        elif state == 'test':
            with torch.no_grad():
                target = unpack_func(label, device)
                output = model(data)
                loss, loss_list = loss_func(output, target, device)

                # Step Logging:
        n_total += n_batch
        loss_total += loss.detach().cpu().numpy() * n_batch
        if metric_func is not None:
            metric = metric_func(output, label)
            metric = metric * n_batch
            metric_total += metric
        if record_output is True:
            output_all.append(list(out_.detach().cpu() for out_ in output))
            label_all.append(list(l_.detach().cpu() for l_ in label))
            # Output Logging
    results = {
        'metric': metric_total / n_total,
        'loss': loss_total / n_total
    }
    if record_output is True:
        results['output'] = output_all
        results['label'] = label_all
    return results


def calculate_mae(out_, y, bc=np.arange(42.5, 82.5)):
    sp = out_.shape
    prob = np.exp(out_.reshape(sp[0:2]))
    x = prob @ bc
    d = x - y
    mae = np.mean(np.abs(d))
    return mae


_, bc = dmu.num2vect(0, [42, 98], 1, 1)
print(len(bc))

## Construct datasets

# Load!
fp_ = osp.join(EXP_DIR, 'subject_train.csv')
df_subject_train = pd.read_csv(fp_)
fp_ = osp.join(EXP_DIR, 'subject_val.csv')
df_subject_val = pd.read_csv(fp_)
fp_ = osp.join(EXP_DIR, 'subject_test0.csv')
df_subject_test0 = pd.read_csv(fp_)
fp_ = osp.join(EXP_DIR, 'subject_test1.csv')
df_subject_test1 = pd.read_csv(fp_)

fp_ = osp.join(EXP_DIR, 'session_train.csv')
df_session_train = pd.read_csv(fp_)
fp_ = osp.join(EXP_DIR, 'session_val.csv')
df_session_val = pd.read_csv(fp_)
fp_ = osp.join(EXP_DIR, 'session_test0.csv')
df_session_test0 = pd.read_csv(fp_)
fp_ = osp.join(EXP_DIR, 'session_test1.csv')
df_session_test1 = pd.read_csv(fp_)

ps = construct_preprocessing({'method': 'pixel_shift',
                              'x_shift': 2,
                              'y_shift': 2,
                              'z_shift': 2})
mr = construct_preprocessing({'method': 'mirror',
                              'probability': 0.5})
avg = construct_preprocessing({'method': 'average'})
crop = construct_preprocessing({'method': 'crop',
                                'nx': 160,
                                'ny': 192,
                                'nz': 160})
fp_list = list(df_session_train.T1_path.values)
label_list = list([age_, ] for age_ in df_session_train.AgeBasedOnClinicalData.values)
dataset_train = Dataset(fp_list, label_list, [mr, ps, avg, crop])

fp_list = list(df_session_val.T1_path.values)
label_list = list([age_, ] for age_ in df_session_val.AgeBasedOnClinicalData.values)
dataset_val = Dataset(fp_list, label_list, [mr, ps, avg, crop])

fp_list = list(df_session_test0.T1_path.values)
label_list = list([age_, ] for age_ in df_session_test0.AgeBasedOnClinicalData.values)
dataset_test0 = Dataset(fp_list, label_list, [avg, crop])

fp_list = list(df_subject_test1.max_cdr_mri_T1_path.values)
label_list = list([age_, ] for age_ in df_session_test1.AgeBasedOnClinicalData.values)
dataset_test1 = Dataset(fp_list, label_list, [avg, crop])

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=10,
    num_workers=4,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=10,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)
test_loader0 = torch.utils.data.DataLoader(
    dataset_test0,
    batch_size=10,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)
test_loader1 = torch.utils.data.DataLoader(
    dataset_test1,
    batch_size=10,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

# Experiments

## Construct Model

for RUN_NAME in ['run_20191206_00', 'run_20191206_01', 'run_20191206_02', 'run_20191206_03']:
    print('************************')
    print(f'{RUN_NAME}')
    print('------------')

    config_path = f'/well/win-biobank/users/jdo465/age_sex_prediction/{RUN_NAME}/config.ini'
    epoch_path = f'/well/win-biobank/users/jdo465/age_sex_prediction/{RUN_NAME}/epoch_best_mae.p'

    age_setup = (42, 96, 1, 1)
    config = dmu.Configurations(config_path)

    model = dmm.model_selector(config)
    model = torch.nn.DataParallel(model, device_ids=[0, ]).cuda()
    model.load_state_dict(torch.load(epoch_path))
    print(f'Pretrained weights loaded from {epoch_path}')

    config.label_info.description = [age_setup, ]

    _, BC = dmu.num2vect(0, [age_setup[0], age_setup[1]], age_setup[2], age_setup[3])
    N_BIN = len(BC)
    print(N_BIN)
    print(BC)

    C_IN = model.module.classifier.conv_6.in_channels
    conv_last = nn.Conv3d(C_IN, N_BIN, [1, 1, 1], bias=True)
    model.module.classifier.conv_6 = conv_last
    model.cuda()


    ## Optimization with two phases

    def metric_func(x, label, bc=BC):
        x = x[0].detach().to('cpu').numpy()
        label = label[0].detach().to('cpu').numpy()
        mae = calculate_mae(x, label, bc=bc)
        return mae


    N_epoch1 = 24
    N_epoch2 = 10

    # Phase One: Optimize last layer
    print('===============================')
    print('Phase One: Optimize last layer')
    print('-------------------------------')
    dmu.tic()
    device = torch.device("cuda")
    unpack_func = dml.UnpackLabel(config.label_info)
    loss_func = dml.LossCombine(config)
    for p in model.module.parameters():
        p.requires_grad = False
    for p in model.module.classifier.conv_6.parameters():
        p.requires_grad = True
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.module.classifier.conv_6.parameters()), lr=1e-1,
                          momentum=config.momentum,
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    print(epoch_path)

    result = go_one_epoch('test', model, loss_func, unpack_func, device, val_loader,
                          metric_func=metric_func, record_output=True)
    dt = dmu.toc(since_last_toc=True, verbose=False)
    print(f"Val-before warm-up: loss={result['loss']:.7f} mae={result['metric']:.3f}, dt={dt:.1f}s")

    for epoch in range(1, N_epoch1 + 1):
        result = go_one_epoch('train', model, loss_func, unpack_func, device, train_loader,
                              metric_func=metric_func, record_output=False, optimizer=optimizer)
        scheduler.step()
        dt = dmu.toc(since_last_toc=True, verbose=False)
        print(
            f"{epoch}: loss={result['loss']:.7f} mae={result['metric']:.3f}, dt={dt:.1f}s, lr={scheduler.state_dict()['_last_lr']}")

    dt = dmu.toc(since_last_toc=True, verbose=False)
    result = go_one_epoch('test', model, loss_func, unpack_func, device, val_loader,
                          metric_func=metric_func, record_output=True)
    print(f"Val-after warm-up: loss={result['loss']:.7f} mae={result['metric']:.3f}, dt={dt:.1f}s")

    # Phase Two: Finetune the full model
    print('===============================')
    print('Phase Two: Finetune the full model')
    print('-------------------------------')

    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    y0 = df_session_val.AgeBasedOnClinicalData.values
    u0 = df_session_val.AgeBasedOnClinicalData.values.mean()
    print(f"randome prediction baseline mae (val)= {dmu.mae(y0 - u0):.3f}")
    y0 = df_session_train.AgeBasedOnClinicalData.values
    u0 = df_session_train.AgeBasedOnClinicalData.values.mean()
    print(f"randome prediction baseline mae (train)= {dmu.mae(y0 - u0):.3f}")

    for epoch in range(1, N_epoch2 + 1):
        result = go_one_epoch('train', model, loss_func, unpack_func, device, train_loader,
                              metric_func=metric_func, record_output=False, optimizer=optimizer)
        scheduler.step()
        dt = dmu.toc(since_last_toc=True, verbose=False)
        print(
            f"{epoch}: loss={result['loss']:.7f} mae={result['metric']:.3f}, dt={dt:.1f}s, lr={scheduler.state_dict()['_last_lr']}")
    result = go_one_epoch('test', model, loss_func, unpack_func, device, val_loader,
                          metric_func=metric_func, record_output=True)
    print(f"val mae = {result['metric']:.3f}")

    fp_ = osp.join(EXP_DIR, f'finetune_{RUN_NAME}_20201219.p')
    torch.save(model, fp_)
    print(f'Model Saved: {fp_}')

# Code for dataset
import numpy as np
import torch
from torch import nn
import sys 
import nibabel as nib

def construct_preprocessing(kwd):
    """
    Method which returns a pre-processing function specified by kwd.
    kwd: dictionary of one of the following forms:
     - {'method': 'pixel_shift',
      'x_shift': int,
      'y_shift': int,
      'z_shift': int} 
            --> returns a method to randomly shift data along the 
                three axis (randomly forth and back with maximum difference x_shift/y_shift/z_shift)
     - {'method': 'mirror',
      'probability': 0.5}
            -->  returns a method which mirrors the MRI scan along the first axis
     - {'method': 'average'}
     - {'method': 'crop',
        'nx': 160,
        'ny': 192,
        'nz': 160}
    """
    if kwd['method'] == 'pixel_shift':

        def pixel_shift(data, x0=kwd['x_shift'], y0=kwd['y_shift'], z0=kwd['z_shift']):
            '''
            data - MRI image - torch.Tensor - shape (n_x,n_y,n_z) - with x-axis is pointing to the right, y-axis 
            to the front (through the nose) and z-axis to the top.
            Function randomly shifts data array.
            Warning: x0,y0,z0 should be chosen such that the offsets are not parts of any non-black voxels.
            '''
            x_shift = int(torch.randint(low=-x0, high=x0 + 1,size=[1]).item())
            y_shift = int(torch.randint(low=-y0, high=y0 + 1,size=[1]).item())
            z_shift = int(torch.randint(low=-z0, high=z0 + 1,size=[1]).item())
            data = np.roll(data, x_shift, axis=0)
            data = np.roll(data, y_shift, axis=1)
            data = np.roll(data, z_shift, axis=2)
            return data

        return pixel_shift

    elif kwd['method'] == 'mirror':

        def mirror(data, p0=kwd['probability']):
            '''
            data - MRI image - torch.Tensor - shape (n_x,n_y,n_z) - with x-axis is pointing to the right, y-axis 
            to the front (through the nose) and z-axis to the top.
            p0 - probability of mirroring the x-axis
            '''
            p = torch.rand(1).item()
            if p < p0:
                data = np.flip(data, 0).copy()
            return data

        return mirror

    elif kwd['method'] == 'average':
        '''
            data - MRI image - torch.Tensor - shape (n_x,n_y,n_z) - with x-axis is pointing to the right, y-axis 
            to the front (through the nose) and z-axis to the top.
            Function divides data by its mean and centers it zero mean, i.e. the resulting data vector has zero mean but variance!=1 possibly.
        '''
        def average(data):
            data = data / np.mean(data)
            return data

        return average

    elif kwd['method'] == 'crop':

        def crop(data, nx=kwd['nx'], ny=kwd['ny'], nz=kwd['nz']):
            '''
            data - MRI image - torch.Tensor - shape (n_x,n_y,n_z) - with x-axis is pointing to the right, y-axis 
            to the front (through the nose) and z-axis to the top.
            '''
            nx0, ny0, nz0 = data.shape
            x_start = np.floor((nx0 - nx) / 2).astype(np.int)
            x_end = nx0-np.ceil((nx0 - nx) / 2).astype(np.int)
            y_start = np.floor((ny0 - ny) / 2).astype(np.int)
            y_end = ny0-np.ceil((ny0 - ny) / 2).astype(np.int)
            z_start = np.floor((nz0 - nz) / 2).astype(np.int)
            z_end = nz0-np.ceil((nz0 - nz) / 2).astype(np.int)

            data = data[x_start:x_end, y_start:y_end, z_start:z_end]
            return data

        return crop

    else:
        raise Exception('method \'' + kwd['method'] + '\' not support')



class MRIDataset(torch.utils.data.Dataset):

    def __init__(self, file_list, label_list, preprocessing=None):
        '''
        file_list - list of paths/to/files of type nii.giz
        label_list - list/array of labels of the same length as the file_list
        preprocessing - a list of functions transforming a 3d MRI scan
        '''
        self.file_list = file_list
        self.label_list = label_list
        self._len = len(file_list)
        self.preprocessing = preprocessing

    def get_data(self, idx):
        label = self.label_list[idx]
        fp_ = self.file_list[idx]
        x = nib.load(fp_).get_fdata()
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
        label = torch.tensor(label, dtype=torch.float32, requires_grad=False)
        return data, label


def give_mri_data(fp_list,label_list,data_type,batch_size=1,num_workers=1,shuffle=True,preprocessing='full'):
    
    #Construct preprocessing functions:
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
	
    #Pick a list of preprocessing functions:
    if preprocessing=='full':
        preproc_train=[avg,crop,ps,mr]
        preproc_val=[avg,crop]

    elif preprocessing=='min':
        preproc_train=[avg,crop]
        preproc_val=[avg,crop]
 
    elif preprocessing=='none':
        preproc_train=[]
        preproc_val=[]
 
    else:
        sys.exit("Unknown preprocessing combination.")

    if data_type=='train':
        data_set = MRIDataset(fp_list, label_list, preproc_train)
    else: 
        data_set = MRIDataset(fp_list, label_list, preproc_val)

    
    batch_size_=min(data_set._len,batch_size)

    #Return data loader:
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size_,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True
    )
    
    return(data_set,data_loader)
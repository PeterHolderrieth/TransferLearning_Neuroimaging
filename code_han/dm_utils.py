import scipy
from scipy.stats import norm
import numpy as np
import configparser
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


CONFIG_PATH = '/well/win-biobank/users/jdo465/age_sex_prediction/run_20190112_00/config.ini'

def get_run_str(python_path, script_path, para):
    'python some_script.py --para_key1=para_value1 --para_key2=para_value2 ...'
    para_str = ''
    for key in para.keys():
        para_str = para_str + f' \\\n--{key}={para[key]}'
    run_str = python_path + ' ' + script_path + para_str
    return run_str

def load_64_parts(filename_pattern, file_part_dir, data_shape):
    """

    :param filename_pattern: e.g. FM_%02d.npy
    :param file_part_dir: directory containing all file parts
    :param data_shape: tuple. 3 elements for 4d files, 4 elements for 5d files.
    :return:
    """
    istart = np.zeros(3, dtype=int)
    iend = np.zeros(3, dtype=int)
    for ipart in range(64):
        fullfilename = filename_pattern%ipart
        p = osp.join(file_part_dir, fullfilename)
        data_part = np.load(p)
        ixyz = np.unravel_index(ipart, [4, 4, 4])
        for j in range(3):
            lj = int(np.ceil(data_shape[-3 + j] / 4))
            istart[j] = lj * ixyz[j]
            iend[j] = lj * (ixyz[j] + 1)

        if ipart == 0:
            if np.ndim(data_part) == 3:
                data = np.zeros(data_shape).astype(np.float32)
            else:
                N = data_part.shape[0]
                data = np.zeros((N, ) + data_shape).astype(np.float32)
        if len(data_shape) == 3:
            if np.ndim(data_part) == 3:
                data[istart[0]:iend[0], istart[1]:iend[1], istart[2]:iend[2]] = data_part.astype(np.float32)
            else:
                data[:, istart[0]:iend[0], istart[1]:iend[1], istart[2]:iend[2]] = data_part.astype(np.float32)
        else:
            data[:, :, istart[0]:iend[0], istart[1]:iend[1], istart[2]:iend[2]] = data_part.astype(np.float32)
    return data


def concat_fmrib_info(sections, subject_no_list=None, var_all=None, config=None):
    if config is None:
        for i, ids in enumerate(sections):
            p = osp.join('/well/win-biobank/users/jdo465/age_sex_prediction/fmrib_info_sections', 'fmrib_info_section_%02d.csv'%ids)
            if i==0:
                info = pd.read_csv(p)
            else:
                info = pd.concat((info, pd.read_csv(p)), sort=False)
    else:
        for i, ids in enumerate(sections):
            import dm_model.dm_dataset as dmd
            subject = dmd.UKBiobankSubjects(config, section_no=ids, dataset_size=config.N_train)
            if i==0:
                info = subject.section_info
            else:
                info = pd.concat((info, subject.section_info), sort=False)
    if subject_no_list is None:
        return info
    else:
        n = info.shape[0]
        m = var_all.shape[-1]
        var_mat = np.zeros([n, m])
        for i, subject_no in enumerate(info.No.values):
            idx = np.argmax(subject_no == subject_no_list)
            var_mat[i] = var_all[idx]
        return info, var_mat

def get_varmat_by_subject_no(subject_no_list, subject_no_list_var_all, var_all):
    n = subject_no_list.shape[0]
    m = var_all.shape[-1]
    var_mat = np.zeros([n, m])
    for i, subject_no in enumerate(subject_no_list):
        idx = np.argmax(subject_no == subject_no_list_var_all)
        var_mat[i] = var_all[idx]
    return var_mat

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers


class Configurations(object):
    config_path = None
    # ExpInfo
    run_name = None
    server_name = None
    exp_config = None
    modality = None
    # ModelInfo
    ModelInfo = None
    N_fc1 = None
    N_fc2 = None
    N_fc3 = None
    selected_model = None
    channel_size = None
    # DataInfo
    dataset = None
    N_train = None
    N_test = None
    id_test = None
    id_train = None
    crop_size = None
    downsample = None
    bin_range = None
    bin_step = None
    label_distribution_sigma = None
    label_info = None
    # Training
    learning_rate = None
    weight_decay = None
    decay_epoch = None
    decay_gamma = None
    momentum = None
    total_epoch = None
    log_interval = None
    batch_size = None
    seed_number = None
    num_workers = None
    scheduler_type = None
    preprocessing = None
    optimizer = None
    # Auxiliary information
    auxiliary_setting = None
    # Path
    path_settings = None
    project_dir = None
    data_dir = None
    save_file_path_folder = None
    run_folder_path = None
    log_dir = None

    def __init__(self,
                 config_path='/well/win-biobank/users/jdo465/age_sex_prediction/run_20181209_00_test/config.ini'):
        self.config_path = config_path
        config = configparser.ConfigParser()
        if not osp.isfile(config_path):
            raise Exception('File %s does not exist' % config_path)
        config.read(config_path)
        # ExpInfo
        exp_info = config['ExpInfo']
        self.run_name = exp_info['run_name']
        self.server_name = exp_info['server_name']
        self.exp_config = exp_info['exp_config']
        self.modality = exp_info['modality']
        # ModelInfo
        model_info = config['ModelInfo']
        self.N_fc1 = int(model_info['N_fc1'])
        self.N_fc2 = int(model_info['N_fc2'])
        self.selected_model = model_info['selected_model']
        self.channel_size = eval(model_info['channel_size'])

        # DataInfo
        data_info = config['DataInfo']
        if 'dataset' in data_info.keys():
            self.dataset = data_info['dataset'].lower()
        else:
            self.dataset = 'ukbiobank'.lower()
        self.N_train = data_info['N_train']
        self.N_test = data_info['N_test']
        self.id_test = eval(data_info['id_test'])
        self.id_train = eval(data_info['id_train'])
        self.crop_size = eval(data_info['crop_size'])
        self.downsample = int(data_info['downsample'])
        self.bin_range = eval(data_info['bin_range'])
        self.bin_step = float(data_info['bin_step'])
        self.label_distribution_sigma = float(data_info['label_distribution_sigma'])
        self.N_fc3 = int((self.bin_range[1] - self.bin_range[0]) / self.bin_step)
        label_info = eval(data_info['label'])
        self.label_info = pd.DataFrame(data=label_info,
                                       columns=('tag',
                                                'type',
                                                'var_id',
                                                'description',
                                                'weights'))
        # Training
        training_info = config['TrainingInfo']
        self.learning_rate = float(training_info['learning_rate'])
        self.weight_decay = float(training_info['weight_decay'])
        self.decay_epoch = int(training_info['decay_epoch'])
        self.decay_gamma = float(training_info['decay_gamma'])
        self.momentum = float(training_info['momentum'])
        self.total_epoch = int(training_info['total_epoch'])
        self.log_interval = int(training_info['log_interval'])
        self.batch_size = int(training_info['batch_size'])
        self.seed_number = int(training_info['seed_number'])
        self.num_workers = int(training_info['num_workers'])
        if 'optimizer' in training_info.keys():
            self.optimizer = training_info['optimizer']
        else:
            self.optimizer = 'sgd'
        # Additional settings
        if 'preprocessing' in training_info.keys():
            self.preprocessing = eval(training_info['preprocessing'])
        if 'scheduler_type' in training_info.keys():
            self.scheduler_type = training_info['scheduler_type']
        else:
            self.scheduler_type = 'step'
        # Auxiliary information
        if 'auxiliary' in exp_info:
            auxiliary_key = exp_info['auxiliary']
            self.auxiliary_setting = config[auxiliary_key]
        # Path
        self.path_settings = eval(config[self.exp_config][self.server_name])
        self.project_dir = self.path_settings['project_dir']
        self.data_dir = self.path_settings['data_dir']
        self.run_folder_path = osp.join(self.project_dir, self.run_name)
        self.log_dir = osp.join(self.run_folder_path, 'logs')


class ScalarRecord(dict):
    """
    Create a recorder. For scalar (0 dim) or vector (n dim, 0th is the concat axis).
    Example:
         logger = ScalarRecord()
         for i in range(10):
            key_value = np.sin(i)
            logger.add('key_value', key_value)
    """
    def __init__(self):
        super(ScalarRecord).__init__()
    def add(self, key, value):
        if not key in self.keys():
            if np.ndim(value) == 0:
                self[key] = np.array((value,))
            else:
                self[key] = value
        else:
            if np.ndim(value) == 0:
                self[key] = np.concatenate((self[key], np.array((value,))))
            else:
                self[key] = np.concatenate((self[key], value), axis=0)
    def plot(self):
        for key in self.keys():
            plt.plot(self[key],label=key)
        plt.grid(True)
        plt.legend()

def mae(d):
    m = np.mean(np.abs(d))
    return m

def pandas_df_to_markdown_table(df, verbose=False):
    from IPython.display import Markdown, display
    fmt = [':---:' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    md_str = df_formatted.to_csv(sep="|", index=False)
    if verbose:
        display(Markdown(md_str))
    return md_str


def tic():
    global t_tic_tag_1945, t_toc_tag_1945_last
    t_tic_tag_1945 = time.time()
    t_toc_tag_1945_last = time.time()

def toc(fmt=f'Elapsed time %.3f seconds', verbose=True, since_last_toc=False):
    global t_tic_tag_1945, t_toc_tag_1945_last
    if since_last_toc is True:
        dt = time.time() - t_toc_tag_1945_last
        t_toc_tag_1945_last = time.time()
    else:
        dt = time.time() - t_tic_tag_1945
    if verbose is True:
        print(fmt%dt)
    return dt


def crop_center(data, n_c):
    nx_c, ny_c, nz_c = n_c
    nx, ny, nz = data.shape
    x_start = np.ceil((nx - nx_c) / 2).astype(np.int)
    x_end = - np.floor((nx - nx_c) / 2).astype(np.int)
    y_start = np.ceil((ny - ny_c) / 2).astype(np.int)
    y_end = - np.floor((ny - ny_c) / 2).astype(np.int)
    z_start = np.ceil((nz - nz_c) / 2).astype(np.int)
    z_end = - np.floor((nz - nz_c) / 2).astype(np.int)
    step = 1
    if nz < nz_c:
        z_start = np.ceil((nz_c - nz) / 2).astype(np.int)
        z_end = - np.floor((nz_c - nz) / 2).astype(np.int)
        data_crop = data[x_start:x_end:step, y_start:y_end:step, ::step]
        data = np.zeros([nx_c, ny_c, nz_c])
        data[:, :, z_start:z_end] = data_crop
    else:
        data = data[x_start:x_end:step, y_start:y_end:step, z_start:z_end:step]
    return data


def plot_training_curv(run_name, project_path='/well/win-biobank/users/jdo465/age_sex_prediction/',
                       labels=['label_00_age/mae', 'label_00_age/loss'], sublabels=['train', 'test'], plot=False):
    from tensorboard.backend.event_processing import event_accumulator
    exp_path = osp.join(project_path, run_name)
    result = pd.DataFrame()
    for label in labels:
        for sublabel in sublabels:
            path_ = osp.join(exp_path, 'logs/', label, sublabel)
            ea = event_accumulator.EventAccumulator(path_)
            ea.Reload()
            df = pd.DataFrame(ea.Scalars(label))
            if plot is True:
                plt.plot(df['step'], df['value'], label='%s/%s'%(run_name,sublabel))
                plt.grid(b=True)
                plt.legend()
                plt.title(label)
            result['step'] = df['step']
            result[label+'/'+sublabel] = df['value']
    return result

def check_and_mkdir(dir_path, mkdir_flag=False, verbose=True):
    is_exist = osp.isdir(dir_path)
    if not is_exist:
        if mkdir_flag:
            os.mkdir(dir_path)
            print(f'Dir made: {dir_path}')
        else:
            print(f'Dir not exist: {dir_path}')
    elif verbose:
        print(f'Selected dir: {dir_path}')
    return is_exist


def check_file(file_path, verbose=True):
    is_file_exist = osp.isfile(file_path)
    if is_file_exist:
        print(f'Exist: {file_path}')
    else:
        print(f'Missing: {file_path}')
    return is_file_exist

import pickle
def pickle_load(file_path, verbose=False):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if verbose is True:
        print(f'File loaded: {file_path}')
    return data


def check_gpu_status():
    import subprocess
    out_ = subprocess.check_output(['cat','/proc/sys/kernel/hostname'])
    out_ = out_.decode('UTF-8')
    print(out_)
    out_ = subprocess.check_output('nvidia-smi')
    out_ = out_.decode('UTF-8')
    print(out_)


def pd_display(df, max_row=1000, max_col=100):
    from IPython.display import display
    with pd.option_context('display.max_rows', max_row, 'display.max_columns', max_col):  # more options can be specified also
        display(df)


def get_time(fmt='%Y-%m-%d_%H:%M:%S', milisecond=True):
    from datetime import datetime
    now = datetime.now()
    ms = now.microsecond
    now = now.strftime(fmt)
    if milisecond is True:
        return f'{now}_{ms:06d}'
    else:
        return f'{now}'


def print_file_info(filename):
    import sys
    print('==========')
    print('# Experiment date and time')
    print(get_time(fmt='%Y-%m-%d %H:%M:%S', milisecond=False))
    print('# Command')
    print(' '.join(sys.argv))
    print('# Script path')
    print(os.path.abspath(filename))
    print('==========')


def cohens_d(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


def inormal(x):
    """
    x: input 2D matrix.
    Dimension: Subject by Features. Normalisation is along 0th axis
    return: inverf normalised x. (Gaussian distribution)

    % Applies a rank-based inverse normal transformation.
    %
    % Usage: Z = inormal(X)
    %            inormal(X,c)
    %            inormal(X,method)
    %            inormal(X,...,quanti)
    %
    % Inputs:
    % X      : Original data. Can be a vector or an array.
    % c      : Constant to be used in the transformation.
    %          Default c=3/8 (Blom).
    % method : Method to choose c. Accepted values are:
    %              'Blom'   (c=3/8),
    %              'Tukey'  (c=1/3),
    %              'Bliss', (c=1/2)  and
    %              'Waerden' or 'SOLAR' (c=0).
    % quanti : All data guaranteed to be quantitative and
    %          without NaN?
    %          This can be a true/false. If true, the function
    %          runs much faster if X is an array.
    %          Default is false.
    %
    % Outputs:
    % Z      : Transformed data.
    %
    % References:
    % * Van der Waerden BL. Order tests for the two-sample
    %   problem and their power. Proc Koninklijke Nederlandse
    %   Akademie van Wetenschappen. Ser A. 1952; 55:453-458
    % * Blom G. Statistical estimates and transformed
    %   beta-variables. Wiley, New York, 1958.
    % * Tukey JW. The future of data analysis.
    %   Ann Math Stat. 1962; 33:1-67.
    % * Bliss CI. Statistics in biology. McGraw-Hill,
    %   New York, 1967.
    %
    % _____________________________________
    % Anderson M. Winkler
    Adapted by Han Peng to python
    """
    if np.ndim(x) != 2:
        raise Exception(f'dim(x)={np.ndim(x)}. Expected: 2')
    c0 = 3.0/8.0 # Default (Blom, 1958)
    c = c0
    xn = np.zeros(x.shape)
    N = x.shape[0]
    for i in range(x.shape[1]):
        xi = x[:, i]
        idx = np.argsort(xi)
        ri = np.argsort(idx)
        p = ((ri+1 - c) / (N - 2 * c + 1))
        xn[:, i] = np.sqrt(2) * scipy.special.erfinv(2 * p - 1)
    return xn


def median_filter(X, size=3, is_1d_sequence=False):
    """
    Input: X~[Sample, Dim0, Dim1, Dim2, ...]
    Each X[i] is a N-dimensional image.
    """
    if (np.ndim(X)==2) and (is_1d_sequence is False):
        raise Exception('X dimension is 2,'+\
            'make sure image dimension is 1, and specify is_1d_sequence=True.')
    X_out = np.zeros(X.shape)
    for i in range(X.shape[0]):
        X_ = scipy.ndimage.median_filter(X[i], size=3)
        X_out[i] = X_
    return X_out

from multiprocessing import Pool


def par_median_filter(X, num_cores=10, chunk=40, verbose=False):
    X_list_chunk = list()
    N = X.shape[0]
    L = int(np.ceil(N/chunk))
    for i in range(chunk):
        i0 = i*L
        i1 = (i+1)*L
        Xi = X[i0:i1]
        X_list_chunk.append(Xi)
    with Pool(processes=num_cores) as p:
        X_out = p.map(median_filter, X_list_chunk)
    X_out = np.concatenate(X_out, axis=0)
    return X_out
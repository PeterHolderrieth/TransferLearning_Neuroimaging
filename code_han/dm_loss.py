import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import dm_model.dm_utils as dmu


def select_loss_function(model, label_distribution_sigma):
    if label_distribution_sigma == 0:
        loss_func_main = F.nll_loss
    else:
        loss_func_main = my_KLDivLoss
    loss_func = lambda x, y: loss_func_with_position(x, y, model, loss_func_main)
    return loss_func


def loss_func_with_position(x, y, model, loss_func_main):
    global correct
    x0 = x[0]
    x1 = x[1]
    B = x0.shape[0]  # Batch size
    auxiliary_loss = model.auxiliary_loss  # weights of different loss terms
    output_sizes = model.output_sizes
    loss_type = model.loss_type

    loss = loss_func_main(x0, y)
    j = 0
    for i in range(len(auxiliary_loss)):
        loss_weight = auxiliary_loss[i]
        output_size = output_sizes[i]
        if loss_weight > 0:
            if loss_type == 'one_hot':
                # Generate one-hot label
                N = np.product(output_size)
                target = generate_one_hot_pos(B, output_size)
                target = torch.tensor(target).to(x1[j].device)
                # Get loss
                loss_auxiliary = F.nll_loss(x1[j].view([N * B, -1]), target.view(N * B))
                pred = x1[j].view([N * B, -1]).max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum().item()
                num_pred = np.product(pred.shape)
                print('loss_func_with_position %d: %d/%d' % (j, correct, num_pred))
            loss += loss_auxiliary
            j = j + 1
    return loss


def generate_one_hot_pos(B, output_size):
    N = np.product(output_size)
    inds = np.arange(N)
    inds = inds.reshape(1, 1, -1)
    target = np.repeat(inds, B, axis=0)
    return target


def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the second input (ind) is the index of the peak position
    c) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    v = (y[:, 0] != -100)
    n = torch.sum(v)
    #print('EFFECTIVE BATCH NUMBER = %d' % n)
    if n == 0:
        loss = x.new_tensor(0, requires_grad=True)
        return loss
    x2 = x[v]
    y2 = y[v]
    if x2.dim() == 5:
        y2 = y2.reshape(y2.shape + (1, 1, 1)).expand_as(x2)
        n0, n1, n2, n3, n4 = x2.shape
        n = n * n2 * n3 * n4
    loss = loss_func(x2, y2) / n.type(y2.type())
    # loss = loss_func(x2, y2)
    #print(loss)
    return loss


class MyNllLoss(object):
    def __init__(self, weight=None):
        self.weight = weight

    def __call__(self, x, y):
        v = (y != -100)
        n = torch.sum(v)
        y2 = y[v]
        x2 = x[v]
        # print('x, x2')
        # print(x.shape)
        # print(x2.shape)
        # print(y2)
        if x2.ndimension() != 2 and y2.ndimension() == 1:
            x2 = x2.reshape((x2.shape[0], x2.shape[1]))
        if n > 0:
            loss = F.nll_loss(x2, y2, reduction='none', weight=self.weight)
            loss = torch.sum(loss)/loss.shape[0]
        else:
            loss = x.new_tensor(0, requires_grad=True)
        return loss


def my_mse_loss(x, y):
    loss_func = torch.nn.MSELoss(reduction='none')
    v = (1 - np.isnan(y)).reshape([-1, ])
    n = torch.sum(v)
    y = y.reshape([-1, ])
    y2 = y[v].reshape([-1, ])
    x2 = x[v].reshape([-1, ])
    """
    print('my_mse_loss')
    for i in range(x.shape[0]):
        print('%.3f, %.3f, %d' % (x[i], y[i], v[i]))
    """
    if n > 0:
        loss = loss_func(x2, y2)
        loss = torch.sum(loss)/loss.shape[0]
    else:
        loss = x.new_tensor(0, requires_grad=True)
    return loss


def my_l1_loss(x, y):
    loss_func = torch.nn.L1Loss(reduction='none')
    v = 1 - np.isnan(y)
    n = torch.sum(v)
    y2 = y[v].reshape([-1, 1])
    x2 = x[v]
    """
    print('my_mse_loss')
    for i in range(x.shape[0]):
        print('%.3f, %.3f, %d' % (x[i], y[i], v[i]))
    """
    if n > 0:
        loss = loss_func(x2, y2)
        loss = torch.sum(loss)/loss.shape[0]
    else:
        loss = x.new_tensor(0, requires_grad=True)
    return loss


class LossFunc(object):
    def __init__(self, label_info: pd.DataFrame):
        self.label_info = label_info
        if label_info.type == 'soft':
            self.loss_func = my_KLDivLoss
        elif label_info.type == 'onehot':
            if len(label_info.description) > 3:
                w = label_info.description[3:]
                w = torch.tensor(w).cuda()
            else:
                w = None
            self.loss_func = MyNllLoss(weight=w)
        elif label_info.type == 'regression':
            if label_info.description == 'L2':
                self.loss_func = my_mse_loss
            if label_info.description == 'L1':
                self.loss_func = my_l1_loss
        elif label_info.type == 'segmentation':
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            raise Exception('Unknown label_info type %s' % label_info.type)

    def __call__(self, x, y):
        #  print(x)
        #  print(y)
        loss = self.loss_func(x, y)
        return loss


class ConvertFunc(object):
    # NOTE: deal with nan: feed with random label
    def __init__(self, label_info):
        self.label_info = label_info

    def __call__(self, x, device):
        label_info = self.label_info
        if label_info.type == 'soft':
            bin_range = label_info.description[0:2]
            bin_step = label_info.description[2]
            sigma = 1
            v, _ = dmu.num2vect(x=x, bin_range=bin_range,
                                bin_step=bin_step, sigma=sigma)
            v[np.isnan(v[:, 0])] = -100
            v = torch.tensor(v, dtype=torch.float32).to(device)
            return v
        elif label_info.type == 'onehot':
            bin_range = label_info.description[0:2]
            # WTF. Be careful with the unwanted link among variables
            v = x.new_tensor(x).reshape([-1, ]).numpy()
            v -= bin_range[0]  # IMPORTANT! Max(v) should be smaller than length of bin_size
            v[np.isnan(v)] = -100
            # print(v, end=', ')
            # print(v_rand)
            v = torch.tensor(v, dtype=torch.int64).to(device)
            return v
        elif label_info.type == 'regression':
            v = torch.tensor(x, dtype=torch.float32).to(device).reshape(-1)
            return v
        elif label_info.type == 'segmentation':
            v = torch.tensor(x, dtype=torch.float32).to(device)
            return v


class UnpackLabel(object):
    def __init__(self, label_info_list: pd.DataFrame):
        self.num_label = len(label_info_list)
        self.convert_func_list = list()
        for i in range(self.num_label):
            label_info = label_info_list.iloc[i]
            convert_func = ConvertFunc(label_info)
            self.convert_func_list.append(convert_func)

    def __call__(self, label, device):
        label_unpacked = list()
        for i in range(self.num_label):
            convert_func = self.convert_func_list[i]
            v = convert_func(label[i], device)
            # print(i, end=', ')
            # print(v.shape, end='; ')
            # print('')
            label_unpacked.append(v)
        return label_unpacked


class LossCombine(object):
    def __init__(self, config: dmu.Configurations):
        self.loss_func_list = list()
        self.num_loss = len(config.label_info)
        self.weights = config.label_info.weights.values
        for i in range(self.num_loss):
            label_info = config.label_info.iloc[i]
            loss_func = LossFunc(label_info)
            # TODO: Remove this part
            """            
            bin_range = label_info.description[0:2]
            bin_step = label_info.description[2]
            sigma = 1
            if label_info.type == 'soft':
                v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                    bin_step=bin_step, sigma=sigma)
                # TODO: remove unnecessary nx/ny
                # TODO dnx = v.shape[1]
                # TODO dny = v.shape[1]
            elif label_info.type == 'onehot':
                v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                    bin_step=bin_step, sigma=sigma)
                dnx = v.shape[1]
                dny = 1
            else:
                raise Exception('Unknown label_info type %s' % label_info.type)
            # TODO self.nx.append(self.nx[i] + dnx)
            # TODO self.ny.append(self.ny[i] + dny)
            """
            self.loss_func_list.append(loss_func)

    def __call__(self, x, y, device):
        loss = torch.tensor(0, dtype=torch.float32).to(device)
        loss_list = list()
        weights_gpu = torch.tensor(self.weights, dtype=torch.float32).to(device)
        for i in range(self.num_loss):
            xi = x[i]
            yi = y[i]
            loss_i = self.loss_func_list[i](xi, yi)
            loss += loss_i*weights_gpu[i]
            loss_list.append(loss_i)
        return loss, loss_list

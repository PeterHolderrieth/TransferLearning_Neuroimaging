import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dm_model.dm_utils as dmu
from .model_files import unet as dm_unet
from .model_files import resnet3d as dm_resnet
import re

def model_selector(config: dmu.Configurations):
    """
    net = model_selector(config: with selected_model, downsample, channel_size) returns a function
    that takes the form
    model =  Net(config)

    for dense_fcn, key words are:
    _%d: layer number to start 1 by 1 by 1 convolution
    optional:
    _avg: with average pooling
    _dropout: with dropout layer before avgpool
    _new: new version of model DenseFcnNew
    """
    selected_model = config.selected_model
    if selected_model == 'naive_fcnn_dropout':
        net = NaiveFcnnDropout(config)
    elif selected_model == 'unet3d':
        net = MyUNet3D(config)
    elif 'resnet' in selected_model:
        net = MyResNet3D(config)
    elif 'dense_fcn' in selected_model:
        # Example: dense_fcn_3_new_dropout_avg_spt
        # avg: average_pooling
        # spt: spatial_prediction
        new_version = ('new' in selected_model)
        multi_head = ('multi' in selected_model)
        if new_version is True:
            net = DenseFcnNew(config)
        elif multi_head is True:
            net = DenseFcnMultiHead(config)
        else:
            net = DenseFcn(config)
    else:
        raise RuntimeError('selected_model undefined!')
    return net

class MyResNet3D(nn.Module):
    def __init__(self, config: dmu.Configurations):
        super(MyResNet3D, self).__init__()
        self.config = config
        selected_model = config.selected_model
        if 'dropout' in selected_model:
            dropout = True
        else:
            dropout = False
        init_channel_number = config.channel_size[0]
        label_info = config.label_info.iloc[0]
        if label_info.type == 'regression':
            nx = 1
        else:
            bin_range = label_info.description[0:2]
            bin_step = label_info.description[2]
            v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                bin_step=bin_step, sigma=1)
            nx = v.shape[1]
        layer_numbers = [18, 34, 50, 101, 152]
        nets = [dm_resnet.resnet18,
                dm_resnet.resnet34,
                dm_resnet.resnet50,
                dm_resnet.resnet101,
                dm_resnet.resnet152]
        self.net = None
        for n_, net_ in zip(layer_numbers, nets):
            if '%d'%n_ in selected_model:
                self.net = net_(num_classes=nx, dropout=dropout)

    def forward(self, x):
        out = (self.net(x), )
        return out

class MyUNet3D(nn.Module):
    def __init__(self, config: dmu.Configurations):
        super(MyUNet3D, self).__init__()
        init_channel_number = config.channel_size[0]
        label_info = config.label_info.iloc[0]
        if label_info.type == 'regression':
            self.nx = 1
        else:
            bin_range = label_info.description[0:2]
            bin_step = label_info.description[2]
            v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                bin_step=bin_step, sigma=1)
            self.nx = v.shape[1]
        self.unet = dm_unet.UNet3D(init_channel_number=init_channel_number)
        self.classifier = nn.Sequential(
            nn.Conv3d(4 * init_channel_number, 4 * init_channel_number, 3, padding=1),
            nn.BatchNorm3d(4 * init_channel_number),
            nn.MaxPool3d(2),
            nn.ReLU(),
            nn.Conv3d(4 * init_channel_number, 8 * init_channel_number, 3, padding=1),
            nn.BatchNorm3d(8 * init_channel_number),
            nn.MaxPool3d(2),
            nn.ReLU(),
            nn.Conv3d(8 * init_channel_number, 8 * init_channel_number, 1, padding=0),
            nn.BatchNorm3d(8 * init_channel_number),
            nn.ReLU(),
            nn.AvgPool3d((4, 5, 4)),
            nn.Dropout(0.5),
            nn.Conv3d(8 * init_channel_number, self.nx, 1, padding=0)
        )
        if not label_info.type == 'regression':
            self.classifier.add_module('logsoftmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        x0, x = self.unet(x)
        x0 = self.classifier(x0).reshape([-1, self.nx])
        return x0, x


class DenseFcn(nn.Module):
    def __init__(self, config: dmu.Configurations):
        super(DenseFcn, self).__init__()
        self.config = config

        # Parse Model Information
        selected_model = config.selected_model
        dense_layer = int(re.findall('\d+', selected_model)[0])
        average_pool = ('avg' in selected_model)
        dropout = ('dropout' in selected_model)
        large_kernel_size = ('lks' in selected_model)

        label_info = config.label_info.iloc[0]
        if label_info.type == 'regression':
            self.f_out = 1
        else:
            bin_range = label_info.description[0:2]
            bin_step = label_info.description[2]
            v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                bin_step=bin_step, sigma=1)
            self.f_out = v.shape[1]

        self.feature_extractor = nn.Sequential()
        for i in range(dense_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = self.config.channel_size[i-1]
            out_channel = self.config.channel_size[i]
            self.feature_extractor.add_module('conv_%d' % i,
                                              self.conv_layer(in_channel,
                                                              out_channel,
                                                              maxpool=True,
                                                              kernel_size=3,
                                                              padding=1))
        self.classifier = nn.Sequential()
        for i in range(dense_layer, 6):
            if i < 6 and large_kernel_size is True:
                ks = 3
                pad = 1
            else:
                ks = 1
                pad = 0
            in_channel = self.config.channel_size[i-1]
            out_channel = self.config.channel_size[i]
            self.classifier.add_module('conv_%d' % i,
                                       self.conv_layer(in_channel,
                                                       out_channel,
                                                       maxpool=False,
                                                       kernel_size=ks,
                                                       padding=pad))

        if average_pool is True:
            if dense_layer == 3:
                avg_shape = [20, 24, 20]
            elif dense_layer == 4:
                avg_shape = [10, 12, 10]
            elif dense_layer == 5:
                avg_shape = [5, 6, 5]
            else:
                raise Exception('Incorrect dense_layer. Expected 3 or 4 or 5, got %d' % dense_layer)
            self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))

        i = 6
        in_channel = self.config.channel_size[i-1]
        out_channel = self.f_out
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return out


class SpatialDropout3D(nn.Module):
    def __init__(self, probs=0.5):
        super(SpatialDropout3D, self).__init__()
        self.distribution_sampler = torch.distributions.Bernoulli(probs=probs)
        self.probs = probs
    def forward(self, data_in):
        if self.training:
            (N, C, X, Y, Z) = data_in.shape
            m = self.distribution_sampler.sample((N, 1, X, Y, Z)) * 1.0 / (1-self.probs)
            m = m.to(data_in.device)
            data_out = data_in * m
            return data_out
        else:
            return data_in


class SpatialFC3D(nn.Module):
    def __init__(self, in_channel, out_channel, x, y, z):
        super(SpatialFC3D, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channel, in_channel, x, y, z))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    def forward(self, data_in):
        data_out = torch.einsum('oixyz,nixyz->noxyz', (self.weight, data_in))
        return data_out


class SpatialCombine(nn.Module):
    def __init__(self, in_positions, out_positions):
        super(SpatialCombine, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_positions, in_positions))
        self.out_positions = out_positions
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    def forward(self, data_in):
        n, c = (data_in.shape[0], data_in.shape[1])
        data_in = data_in.reshape([n, c, -1])
        w = self.weight/torch.sum(self.weight, dim=1, keepdim=True)
        data_out = torch.einsum('oi,nci->nco', (w, data_in))
        data_out = data_out.reshape([n, c, self.out_positions, 1, 1])
        return data_out


class DenseFcnNew(nn.Module):
    def __init__(self, config: dmu.Configurations):
        super(DenseFcnNew, self).__init__()
        self.config = config

        # Parse Model Information
        selected_model = config.selected_model
        dense_layer = int(re.findall('\d+', selected_model)[0])
        average_pool = ('avg' in selected_model)
        spatial_dropout = ('spdrop' in selected_model)
        dropout = ('dropout' in selected_model)
        self.spatial_prediction = ('spt' in selected_model)
        self.spatial_fc_prediction = ('spfc' in selected_model)
        self.spatial_combine = ('spcb' in selected_model)
        self.double_fm = ('dfm' in selected_model)
        self.large_kernel_size = ('lks' in selected_model)
        if 'instancenorm' in selected_model:
            self.bn = 'instancenorm'
        elif 'layernorm' in selected_model:
            self.bn = 'layernorm'
        else:
            self.bn = 'batchnorm'
        label_info = config.label_info.iloc[0]
        if label_info.type == 'regression':
            self.f_out = 1
        else:
            bin_range = label_info.description[0:2]
            bin_step = label_info.description[2]
            v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                bin_step=bin_step, sigma=1)
            self.f_out = v.shape[1]

        self.feature_extractor = nn.Sequential()
        for i in range(dense_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = self.config.channel_size[i-1]
            out_channel = self.config.channel_size[i]
            if i < dense_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1,
                                                                  bn=self.bn))
            else:
                if self.double_fm is True:
                    s_ = 1
                else:
                    s_ = 2
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1,
                                                                  maxpool_stride=s_,
                                                                  bn=self.bn))

        for i in range(dense_layer, 6):
            in_channel = self.config.channel_size[i-1]
            out_channel = self.config.channel_size[i]
            if i < 6 and self.large_kernel_size is True:
                ks = 3
                pad = 1
            else:
                ks = 1
                pad = 0
            self.feature_extractor.add_module('conv_%d' % i,
                                       self.conv_layer(in_channel,
                                                       out_channel,
                                                       maxpool=False,
                                                       kernel_size=ks,
                                                       padding=pad,
                                                       bn=self.bn))

        self.classifier = nn.Sequential()
        if spatial_dropout is True:
            self.classifier.add_module('spatial_dropout', SpatialDropout3D(0.5))

        if self.double_fm is True:
            if config.crop_size[0]==160:
                if dense_layer == 3:
                    avg_shape = [39, 47, 39]
                elif dense_layer == 4:
                    avg_shape = [19, 23, 19]
                elif dense_layer == 5:
                    avg_shape = [9, 11, 9]
                else:
                    raise Exception('Incorrect dense_layer. Expected 3 or 4 or 5, got %d' % dense_layer)
            elif config.crop_size[0] == 80:
                if dense_layer == 3:
                    avg_shape = [19, 23, 19]
                elif dense_layer == 4:
                    avg_shape = [9, 11, 9]
                else:
                    raise Exception('Incorrect dense_layer. Expected 3 or 4 for input size 80, got %d' % dense_layer)
        else:
            if config.crop_size[0] == 160:
                if dense_layer == 3:
                    avg_shape = [20, 24, 20]
                elif dense_layer == 4:
                    avg_shape = [10, 12, 10]
                elif dense_layer == 5:
                    avg_shape = [5, 6, 5]
                else:
                    raise Exception('Incorrect dense_layer. Expected 3 or 4 or 5, got %d' % dense_layer)
            elif config.crop_size[0] == 80:
                if dense_layer == 3:
                    avg_shape = [10, 12, 10]
                elif dense_layer == 4:
                    avg_shape = [5, 6, 5]
                else:
                    raise Exception('Incorrect dense_layer. Expected 3 or 4 for input size 80, got %d' % dense_layer)

        if 'three' in config.selected_model:
            if dense_layer == 3:
                avg_shape = [12, 16, 12]
            elif dense_layer == 4:
                avg_shape = [6, 8, 6]
            elif dense_layer == 5:
                avg_shape = [3, 4, 3]
            else:
                raise Exception('Incorrect dense_layer. Expected 3 or 4 or 5, got %d' % dense_layer)

        if 'half' in config.selected_model:
            avg_shape[0] //= 2
        if self.spatial_combine is True:
            c_spatial = 1
            self.classifier.add_module('spatial_combine', SpatialCombine(np.prod(avg_shape), c_spatial))
            #self.classifier.add_module('average_pool', nn.AvgPool3d([c_spatial, 1, 1]))
        elif average_pool is True:
            self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))

        n = len(self.config.channel_size)
        if n > 6:
            for i in range(6,n):
                in_channel = self.config.channel_size[i-1]
                out_channel = self.config.channel_size[i]
                self.classifier.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0,
                                                                  bn=self.bn))

        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))

        i = n
        in_channel = self.config.channel_size[i - 1]
        out_channel = self.f_out
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))



        if self.spatial_fc_prediction is True:
            self.spatial_classifier = SpatialFC3D(self.config.channel_size[i-1],
                                                  self.f_out,
                                                  avg_shape[0],
                                                  avg_shape[1],
                                                  avg_shape[2])

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2, bn='batchnorm'):
        if bn == 'instancenorm':
            bn_layer = nn.InstanceNorm3d
        elif bn == 'layernorm':
            bn_layer = nn.LayerNorm
        else:
            bn_layer = nn.BatchNorm3d
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                bn_layer(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                bn_layer(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        if self.spatial_prediction:
            x = self.classifier.conv_6(x_f)
            x = F.log_softmax(x, dim=1)
            out.append(x)
        if self.spatial_fc_prediction:
            x = self.spatial_classifier(x_f)
            x = F.log_softmax(x, dim=1)
            out.append(x)
        return out


class DenseFcnMultiHead(nn.Module):
    def __init__(self, config: dmu.Configurations):
        super(DenseFcnMultiHead, self).__init__()
        self.config = config

        # Parse Model Information
        selected_model = config.selected_model
        label_info = config.label_info.iloc[0]
        if label_info.type == 'regression':
            self.f_out = 1
        else:
            bin_range = label_info.description[0:2]
            bin_step = label_info.description[2]
            v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                bin_step=bin_step, sigma=1)
            self.f_out = v.shape[1]
        self.base_feature_extractor_0 = nn.Sequential()
        for i in range(3):
            in_channel = 1 if i == 0 else self.config.channel_size[i-1]
            out_channel = self.config.channel_size[i]
            self.base_feature_extractor_0.add_module('conv_%d' % i,
                                              self.conv_layer(in_channel,
                                                              out_channel,
                                                              maxpool=(i<2),
                                                              kernel_size=3,
                                                              padding=1,
                                                              maxpool_stride=2))
        i = 3
        self.base_feature_extractor_1 = nn.Sequential()
        in_channel = self.config.channel_size[i-1]
        out_channel = self.config.channel_size[i]
        self.base_feature_extractor_1.add_module('mp_%d'%(i - 1),
                                          nn.MaxPool3d(2, stride=2))
        self.base_feature_extractor_1.add_module('conv_%d' % i,
                                          self.conv_layer(in_channel,
                                                          out_channel,
                                                          maxpool=False,
                                                          kernel_size=3,
                                                          padding=1))
        # classifier 5
        i = 4
        self.classifier5 = nn.Sequential()
        self.classifier5.add_module('mp_%d'%(i - 1),
                                                 nn.MaxPool3d(2, stride=2))
        for i in range(4, 6):
            in_channel = self.config.channel_size[i - 1]
            out_channel = self.config.channel_size[i]
            self.classifier5.add_module('conv_%d' % i,
                                                 self.conv_layer(in_channel,
                                                                 out_channel,
                                                                 maxpool=(i==4),
                                                                 kernel_size=3,
                                                                 padding=1,
                                                                 maxpool_stride=2))
        i = 6
        in_channel = self.config.channel_size[i - 1]
        out_channel = self.f_out
        avg_shape = [5, 6, 5]
        self.classifier5.add_module('average_pool', nn.AvgPool3d(avg_shape))
        self.classifier5.add_module('dropout', nn.Dropout(0.5))
        self.classifier5.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))
        # classifier 4
        i = 4
        self.classifier4 = nn.Sequential()
        self.classifier4.add_module('mp_%d'%(i - 1),
                                    nn.MaxPool3d(2, stride=1))
        for i in range(4, 6):
            in_channel = self.config.channel_size[i - 1]
            out_channel = self.config.channel_size[i]
            self.classifier4.add_module('conv_%d' % i,
                                              self.conv_layer(in_channel,
                                                              out_channel,
                                                              maxpool=False,
                                                              kernel_size=1,
                                                              padding=0))
        i = 6
        in_channel = self.config.channel_size[i - 1]
        out_channel = self.f_out
        avg_shape = [19, 23, 19]
        self.classifier4.add_module('average_pool', nn.AvgPool3d(avg_shape))
        self.classifier4.add_module('dropout', nn.Dropout(0.5))
        self.classifier4.add_module('conv_%d' % i,
                                    nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))
        # Classifier 3
        i = 3
        self.classifier3 = nn.Sequential()
        self.classifier3.add_module('mp_%d'%(i - 1),
                                    nn.MaxPool3d(2, stride=1))
        for i in range(3, 6):
            in_channel = self.config.channel_size[i - 1]
            out_channel = self.config.channel_size[i]
            self.classifier3.add_module('conv_%d' % i,
                                              self.conv_layer(in_channel,
                                                              out_channel,
                                                              maxpool=False,
                                                              kernel_size=1,
                                                              padding=0))
        i = 6
        in_channel = self.config.channel_size[i - 1]
        out_channel = self.f_out
        avg_shape = [39, 47, 39]
        self.classifier3.add_module('average_pool', nn.AvgPool3d(avg_shape))
        self.classifier3.add_module('dropout', nn.Dropout(0.5))
        self.classifier3.add_module('conv_%d' % i,
                                    nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x2 = self.base_feature_extractor_0(x)
        x3 = self.base_feature_extractor_1(x2)

        x_score_5 = self.classifier5(x3)
        x = F.log_softmax(x_score_5, dim=1)
        out.append(x)
        x_score_4 = self.classifier4(x3)
        x = F.log_softmax(x_score_4, dim=1)
        out.append(x)
        x_score_3 = self.classifier3(x2)
        x = F.log_softmax(x_score_3, dim=1)
        out.append(x)

        x_score_avg = (x_score_5 + x_score_4 + x_score_3)/3
        x = F.log_softmax(x_score_avg, dim=1)
        out.insert(0, x)
        return out

class NaiveFcnnDropout(nn.Module):
    def __init__(self,
                 config: dmu.Configurations):
        super(NaiveFcnnDropout, self).__init__()

        cs = config.channel_size

        self.config = config

        self.conv0 = nn.Conv3d(1, cs[0], 3, padding=0)
        self.bn0 = nn.BatchNorm3d(cs[0])

        self.conv1 = nn.Conv3d(cs[0], cs[1], 3, padding=0)
        self.bn1 = nn.BatchNorm3d(cs[1])

        self.conv2 = nn.Conv3d(cs[1], cs[2], 3, padding=0)
        self.bn2 = nn.BatchNorm3d(cs[2])

        self.conv3 = nn.Conv3d(cs[2], cs[3], 3, padding=0)
        self.bn3 = nn.BatchNorm3d(cs[3])

        self.conv4 = nn.Conv3d(cs[3], cs[4], 3, padding=1)
        self.bn4 = nn.BatchNorm3d(cs[4])

        self.conv5 = nn.Conv3d(cs[4], cs[5], 1, padding=0)
        self.bn5 = nn.BatchNorm3d(cs[5])

        self.dropout = nn.Dropout(0.5)

        # >> TODO: multi task, automated building a few blocks
        self.num_loss = len(config.label_info)
        self.classifier = nn.ModuleList()
        self.nx = list()
        for i in range(self.num_loss):
            label_info = config.label_info.iloc[i]
            if label_info.type == 'regression':
                nx = 1
            else:
                bin_range = label_info.description[0:2]
                bin_step = label_info.description[2]
                v, _ = dmu.num2vect(x=np.ones([1, 1]), bin_range=bin_range,
                                    bin_step=bin_step, sigma=1)
                nx = v.shape[1]
            classifier = nn.Conv3d(cs[5], nx, 1, padding=0)
            self.classifier.append(classifier)
            self.nx.append(nx)

        #  End modification.

    def forward(self, x):
        # print('Input')
        # print(x.size())
        # Size should be 158,190,158
        x = F.relu(F.max_pool3d(self.bn0(self.conv0(x)), 2))
        # Size should be 78,94,78
        x = F.relu(F.max_pool3d(self.bn1(self.conv1(x)), 2))
        # Size should be 38,46,38
        x = F.relu(F.max_pool3d(self.bn2(self.conv2(x)), 2))
        # Size should be 18,22,18
        x = F.relu(F.max_pool3d(self.bn3(self.conv3(x)), 2))
        # Size should be 8,10,8
        x = F.relu(F.max_pool3d(self.bn4(self.conv4(x)), 2))
        # Size should be 4,5,4
        x = F.avg_pool3d(F.relu(self.bn5(self.conv5(x))), (4, 5, 4))
        x = self.dropout(x)
        # Size should be 1,1,1
        out = list()
        for i in range(self.num_loss):
            y = self.classifier[i](x)
            y = y.view((-1, self.nx[i]))
            if not self.config.label_info.iloc[i].type == 'regression':
                y = F.log_softmax(y, dim=1)
            out.append(y)
        return out


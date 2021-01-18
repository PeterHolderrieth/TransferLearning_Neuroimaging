import torch
import torch.nn as nn
import torch.nn.functional as F

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True,batch_norm=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.output_dim=output_dim

        #Define part 1: feature extractor
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1,
                                                                  batch_norm=batch_norm))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0,
                                                                  batch_norm=batch_norm))
        #Define part 2: classifier
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))

        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))

        in_channel = channel_number[-1]
        out_channel = self.output_dim
        self.classifier.add_module('conv_%d' % n_layer,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))
        
    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2,batch_norm=True):
        if batch_norm:
            batch_norm_layer=nn.BatchNorm3d(out_channel)
        else: 
            batch_norm_layer=nn.Identity()
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                batch_norm_layer,
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                batch_norm_layer,
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        '''
        x - torch.Tensor - shape (batch_size,1,160,192,160)
        Returns list of size 1 with output torch.Tensor of shape (batch_size,self.output_dim)
        Every row are the log-probabilities of a probability distribution (later the distribution
        over age bins)
        '''
        #out = list()
        batch_size=x.size(0)
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x=x.view(batch_size,-1)
        x = F.log_softmax(x, dim=1)
        return(x)
        #out.append(x)
        #return out

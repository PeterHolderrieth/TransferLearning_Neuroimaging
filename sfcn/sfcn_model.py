import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=0.5,batch_norm=True):
        '''
            channel_number - list of ints - number of channels for 3d convolutions
            output_dim - output dimension of model - e.g. for sex prediction 2 or age prediction the number of bins
            dropout - float - probability of a single node to dropout at final layer
            batch_norm - bool - indicates whether batch normalization is applied
        '''
        super(SFCN, self).__init__()
        self.n_layer = len(channel_number)
        self.output_dim=output_dim
        self.channel_number=channel_number 
        
        #Define part 1: feature extractor
        self.feature_extractor = nn.Sequential()
        for i in range(self.n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < self.n_layer-1:
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

        if dropout is not None:
            self.classifier.add_module('dropout', nn.Dropout(dropout))

        in_channel = channel_number[-1]
        out_channel = self.output_dim
        self.classifier.add_module('conv_%d' % self.n_layer,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))
        
    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2,batch_norm=True):
        '''
        A function to build a convolutional layer including batch normalization, pooling and ReLU activation
        Input:
            in_channel, out_channel - ints - number of in and out channels
            maxpool - bool - indicates whether max-pooling is applied 
            kernel_size.padding,maxpool_stride - int - parameters for model
            batch_norm - bool - indicates whether batch norm is applied
        Output:
            torch.nn.Sequential module - layer 
        '''

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
        Input: x - torch.Tensor - shape (batch_size,1,160,192,160)
        Output: torch.Tensor - shape (batch_size,self.output_dim) 
                    - Every row consists of log-probabilities of a probability distribution.
        '''
        batch_size=x.size(0)
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x=x.view(batch_size,-1)
        x = F.log_softmax(x, dim=1)
        return(x)

    def set_grad_classifier(self, val):
        '''
        Function to set the gradient parameters of classifier.
        Input: val - bool - value of gradient computation of classifier
        '''
        for p in self.classifier.parameters():
            p.requires_grad=val
        
    def set_grad_featext(self,vals,layers):
        '''
        Function to set the gradient parameters of feature extractor.
        Input:  val - list of bools - gradient parameters
                layers - list of ints - index of layer to set gradient parameter
        '''
        if len(layers)!=len(vals):
            sys.exit("Error in set_feature_extractor: Different lengths of val and layers.")
        
        for it in range(len(layers)):
            for p in self.feature_extractor[layers[it]].parameters():
                p.requires_grad=vals[it]
    
    
    def train_final_layer(self):
        '''
        Function to train classifier.
        '''
        self.set_grad_classifier(True)

    def train_full_model(self):
        '''
        After calling this function, all parameters
        of the model will be trained.
        '''
        for p in self.parameters():
            p.requires_grad=True

    def train_nothing(self):
        '''
        After calling this function, no parameters
        of the model will be trained.
        '''
        for p in self.parameters():
            p.requires_grad=False

    def train_only_final_layers(self,n_train_layers:int):
        '''
        Function to train (only!) the final layers of a network.
        Input: n_train_layers - int - number of final layers to train
                                (classifier is last layer)
        '''
        #Set gradient to zero everywhere:   
        self.train_nothing()     

        #Set gradient to true for the final layers:
        if n_train_layers>0:
            self.set_grad_classifier(True)

        if n_train_layers>1:
            n_feature_layers=n_train_layers-1
            
            #Count down from the end:
            feat_layers=[self.n_layer-it-1 for it in range(n_feature_layers)]
            #Set all true: 
            feat_vals=[True for _ in feat_layers]

            self.set_grad_featext(feat_vals,feat_layers)

    
    def reinit_classifier_pres_scale(self):
        '''
        Function to reinitialize the parameters of classifier while preserving the scaling.
        '''
        for p in self.classifier.parameters():
            p_=p.data.detach()
            p.data=p_.std()*torch.randn(size=p_.shape).to(p_.device)+p_.mean()

    def reinit_featext_pres_scale(self,layers=None):
        '''
        Function to reinitialize the parameters of feature extractor while preserving the scaling.
        Input: layers - list of ints - indices of layers to reinitalize - if None everything is reinitialized
        '''
        if layers is None:
            layers=[it for it in range(self.n_layer)]
        for layer in layers:
            for p in self.feature_extractor[layer].parameters():
                p_=p.data.detach()
                p.data=p_.std()*torch.randn(size=p_.shape).to(p_.device)+p_.mean()

    def reinit_final_layers_pres_scale(self,n_train_layers:int):
        '''
        Function to reinitialize the final layers of the network while preserving the scaling.
        Input: n_train_layers - int - number of final layers to train
        '''
        #Set gradient to true for the final layers:
        if n_train_layers>0:
            self.reinit_classifier_pres_scale() 
        if n_train_layers>1:
            n_feature_layers=n_train_layers-1
            feat_layers=[self.n_layer-it-1 for it in range(n_feature_layers)] 
            self.reinit_featext_pres_scale(feat_layers)
    

    def reinit_full_model_pres_scale(self):
        '''
        Function to reinitialize the full model while preserving scaling.
        '''
        self.reinit_classifier_pres_scale() 
        self.reinit_featext_pres_scale()
    
    def change_output_dim(self, new_output_dim: int):
        '''
        new_output_dim - new output dimension
        '''
        c_in = self.classifier.conv_6.in_channels
        conv_last = nn.Conv3d(c_in, new_output_dim, kernel_size=1)
        self.classifier.conv_6 = conv_last
        self.output_dim=new_output_dim



# model=SFCN(dropout=0.5)
# #for name, module in model.named_modules():
# #    print(name)
# print(list(model.feature_extractor[0].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[1].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[2].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[3].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[4].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[5].parameters())[0].flatten()[0])
# print(list(model.classifier.parameters())[0].flatten()[0])
# # print()
# model.module.train_only_final_layers(2)
# model.reinit_full_model_pres_scale()
# print(list(model.feature_extractor[0].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[1].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[2].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[3].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[4].parameters())[0].flatten()[0])
# print(list(model.feature_extractor[5].parameters())[0].flatten()[0])
# print(list(model.classifier.parameters())[0].flatten()[0])

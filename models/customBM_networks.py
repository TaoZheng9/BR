#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

# delete sigmoid and *2 -1 , so this is not activation in the last layer
# for Binaural Ratio the Mag part need  indendpendent 2 layer, not just the last layer. 

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, sigmoid=True, relu=False, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
        print("batch in conv1x1")
    if(sigmoid):
        model.append(nn.Sigmoid())
        print("sigmoid in conv1x1")
    if(relu):
        model.append(nn.ReLU())
        print("relu in conv1x1")
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        print("init 0.02 var conv")
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

def kaiming_weights_init(net):
    print("kaiming init for conv2d and batch norm2d")
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
def small_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
        print("init small var 0.002 conv")
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
import math
class VisualNet(nn.Module):
    def __init__(self, original_resnet,cluster=False,D=1):
        super(VisualNet, self).__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1
        self.cluster=cluster
        self.D=D

    def Initcluster(self):
        D=self.D
        L=int((math.log(D)/math.log(2)))
        if self.cluster==True:
            feature_middle=self.feature_extraction[0:-1-L]
            # print(self.feature_extraction[-1][0].conv1) correct
            # layers=list((feature_middle,self.feature_extraction[-1][0].conv1,self.feature_extraction[-1][0].bn1,self.feature_extraction[-1][0].relu))
            #layers=list((feature_middle,self.feature_extraction[-1-L][0],self.feature_extraction[-1-L][1].conv1,self.feature_extraction[-1-L][1].bn1,self.feature_extraction[-1-L][1].relu))
            layers=list((feature_middle,self.feature_extraction[-1-L][0]))
            self.feature_middle=nn.Sequential(*layers)

    def forward(self, x):
        y = self.feature_extraction(x)
        if self.cluster==True:
            m=self.feature_middle(x)
            return y,m
        else:
            return y

class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2,sigmoid=True,relu=False,batch_norm=True,vis=False,cluster=False):
        super(AudioNet, self).__init__()
        self.vis=vis
        self.cluster=cluster
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(1296, ngf * 8) #1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer4m = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        self.audionet_upconvlayer5m = unet_upconv(ngf * 2, 1, True) # independent uconv now  input and output channel
        self.conv1x1 = create_conv(512, 8, 1, 0, sigmoid=sigmoid,relu=relu,batch_norm=batch_norm) #reduce dimension of extracted visual features

    def forward(self, x, ori_visual_feat):
        if self.cluster==True:
            ori_visual_feat,m=ori_visual_feat
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat8 = self.conv1x1(ori_visual_feat)
        if self.cluster==True:
            return visual_feat8,m
        elif self.vis==True:
            return visual_feat8
        visual_feat = visual_feat8.view(visual_feat8.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature
        
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature  = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        # print("shape:",audio_upconv4feature.shape)
        ratio_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        # print("shape:",ratio_prediction.shape)
        audio_upconv4mfeature = self.audionet_upconvlayer4m(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        # print("shape:",audio_upconv4mfeature.shape)
        mag_prediction = self.audionet_upconvlayer5m(torch.cat((audio_upconv4mfeature, audio_conv1feature), dim=1))
        # print("shape:",mag_prediction.shape)
        return ratio_prediction,mag_prediction




'''
'''
import torchvision
original_resnet = torchvision.models.resnet18(True)
net = VisualNet(original_resnet,cluster=True)
print(net)
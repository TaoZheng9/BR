#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from .customBM_networks import VisualNet, AudioNet, weights_init

class ModelBuilder():
    # builder for visual stream
    def build_visual(self, weights='',pretrained = True,cluster=False,D=1):
        original_resnet = torchvision.models.resnet18(pretrained)
        net = VisualNet(original_resnet,cluster=cluster,D=D)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
            if cluster==True:
                net.Initcluster()
        else:
            print("pretrained:",pretrained)
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights='',relu=False,sigmoid=True,batch_norm=True,vis=False,cluster=False):
        #AudioNet: 5 layer UNet
        print("input_nc,output_nc:",input_nc,output_nc)
        net = AudioNet(ngf, input_nc, output_nc,relu=relu,sigmoid=sigmoid,batch_norm=batch_norm,vis=vis,cluster=cluster)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net

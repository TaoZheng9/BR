#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import customBM_networks,criterion
from torch.autograd import Variable

eps=1e-10

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_visual, self.net_audio = nets
        self.vis=self.net_audio.vis
        print("visualization:",self.vis)

    def forward(self, input, volatile=False):
        visual_input = input['frame']
        audio_target = input['audio_target_spec'] # audio_target_spec
        audio_input = input['audio_input_spec'] # audio_input_spec
        audio_gt = Variable(audio_target[:,:,:-1,:], requires_grad=False) # the last one
        ratio_gt = Variable(input['audio_ratio_target_spec'][:,:,:-1,:], requires_grad=False)

        input_spectrogram = Variable(audio_input, requires_grad=False, volatile=volatile)
        magWeight = Variable((input_spectrogram[:,0]**2+input_spectrogram[:,1]**2).unsqueeze(1)**0.5, requires_grad=False, volatile=volatile)
        visual_feature = self.net_visual(Variable(visual_input, requires_grad=False, volatile=volatile))
        if self.vis==True:
            return self.net_audio(magWeight, visual_feature)
        ratiophase_network_output,mag_network_output = self.net_audio(input_spectrogram, visual_feature)
        ratiophase_prediction=torch.tanh(ratiophase_network_output)
        ratioMag_pred=20*torch.sigmoid(mag_network_output[:,0:])# B1FT
        
        #print("network OK")


        
        # the mag part
        ratioMag_gt=((ratio_gt[:,1]**2+ratio_gt[:,0]**2).unsqueeze(1)**0.5)
        BM_gt=Variable(ratioMag_gt*magWeight[:,:,:-1,:],requires_grad=False)
        BM_pred=ratioMag_pred*magWeight[:,:,:-1,:] # B 1 FT  * B 
        
        #print("ratio mag OK")
        
        #the ratio phase part
        ratiophase_weight_real = magWeight[:,0,:-1,:] * ratiophase_prediction[:,0,:,:] 
        ratiophase_weight_img = magWeight[:,0,:-1,:] * ratiophase_prediction[:,1,:,:] 
        predicted_ratiophase = torch.cat((ratiophase_weight_real.unsqueeze(1), ratiophase_weight_img.unsqueeze(1)), 1)
        tanh_ratioMag_gt=torch.tanh(ratioMag_gt)
        ratiophase_gt=Variable(ratio_gt*tanh_ratioMag_gt/(ratioMag_gt+eps)*magWeight[:,:,:-1,:],requires_grad=False)
        

        #print("ration phase OK")
        
        # the gt part
        # make the ratio_predition
        ratiophase_predictionAbs=Variable((ratiophase_prediction[:,0,]**2+ratiophase_prediction[:,1,]**2).unsqueeze(1)**0.5,requires_grad=False)
        ratio_prediction=ratiophase_prediction*ratioMag_pred/(ratiophase_predictionAbs+eps)
        spectrogram_pred_real = input_spectrogram[:,0,:-1,:] * ratio_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * ratio_prediction[:,1,:,:]
        spectrogram_pred_img = input_spectrogram[:,0,:-1,:] * ratio_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * ratio_prediction[:,0,:,:]
        predicted_spectrogram = torch.cat((spectrogram_pred_real.unsqueeze(1), spectrogram_pred_img.unsqueeze(1)), 1)

        #print("ratio gt OK")
        
        output =  {'ratio_prediction': ratio_prediction, 'ratio_gt': ratio_gt,
                   'pred_ratio_phase': predicted_ratiophase, 'ratio_phase_gt':ratiophase_gt, 
                   'predicted_spectrogram': predicted_spectrogram, 'audio_gt': audio_gt,
                   'BM_gt':BM_gt,'BM_pred':BM_pred,'ratioMag_pred':ratioMag_pred}
        return output

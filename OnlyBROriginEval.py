#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.customBM_models import ModelBuilder
from models.modelRatioBROrigin2in_audioVisual import AudioVisualModel
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import numpy as np


#used to display validation loss
def display_val(model, loss_criterion, writer, dataset_val, opt):
    batch_loss = []
    batch_mse_loss = []
    batch_bm_loss = []
    batch_phase_loss = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                mse_loss = loss_criterion(output['predicted_spectrogram'], output['audio_gt'])
                loss=mse_loss # mse_loss+
                batch_loss.append(loss.item())
                batch_mse_loss.append(mse_loss.item())
            else:
                break
    avg_loss = sum(batch_loss)/len(batch_loss)
    avg_mse_loss = sum(batch_mse_loss)/len(batch_mse_loss)
    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss)
        writer.add_scalar('data/mse_loss', avg_mse_loss)
    print('val loss: %.3f' % avg_loss)
    print('val mse loss: %.3f' % avg_mse_loss)
    return avg_loss


def display_otherMetric(model, writer, dataset_val, opt):
    eps=1e-8
    eps2=1e-4
    magBiases = []
    logMagBiases = []
    magBiases2 = []
    logMagBiases2 = []
    magPhaseErrors=[]
    logMagPhaseErrors=[]
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                # B C F T
                #print("shape of output:",output['BM_pred'].shape,output['BM_gt'].shape)
                one=torch.ones(1).to(opt.device)
                magM=((val_data['audio_input_spec'][:,0:1]**2+val_data['audio_input_spec'][:,1:]**2)[:,:,:-1]**0.5).to(opt.device)# input  0:1 1: prevent B C F t to be BFT
                #print("max min magM",magM.max(),magM.min())
                logMagM=torch.log1p(magM)
                magHatD=(output['predicted_spectrogram'][:,0:1]**2+output['predicted_spectrogram'][:,1:]**2)**0.5
                magGtD=(output['audio_gt'][:,0:1]**2+output['audio_gt'][:,1:]**2)**0.5
                #print("shape of mag:",magM.shape,magHatD.shape,magGtD.shape)
                #print("device:",one.device,magM.device,magHatD.device,magGtD.device)
                FtBias=torch.zeros_like(magHatD)
                FtBias[magHatD<=magGtD]=(one-magHatD/magGtD)[magHatD<=magGtD]
                FtBias[magHatD>magGtD]=(magGtD/magHatD-one)[magHatD>magGtD]

                logMagBias=torch.sum(logMagM*(FtBias),dim=[2,3])/torch.sum(logMagM,dim=[2,3])
                #print("shape of weighted average bias",logMagBias.shape)
                logMagBias=torch.mean(logMagBias)
                #print("shape of mean of weighted average bias",logMagBias.shape)
                magBias=torch.sum(magM*(FtBias),dim=[2,3])/torch.sum(magM,dim=[2,3])
                magBias=torch.mean(magBias)


                magBiases.append(magBias.item()) 
                logMagBiases.append(logMagBias.item())


                # for the phase error
                logMagM=logMagM.cpu().numpy()
                magM=magM.cpu().numpy()
                # new version:  ipd first then ipd error  and error weight on |M|
                LpR=val_data['audio_input_spec'][:,:,:-1].to(opt.device)
                tL=LpR+output['predicted_spectrogram'] # M + MB
                tR=LpR-output['predicted_spectrogram']
                L=LpR+output['audio_gt']
                R=LpR-output['audio_gt']
                tL=tL.cpu().numpy()
                tR=tR.cpu().numpy()
                L=L.cpu().numpy()
                R=R.cpu().numpy()

                #print(LpR.shape,output['binaural_spectrogram'].shape)

                tL_angle=np.angle(tL[:,0:1]+tL[:,1:]*1j)
                L_angle=np.angle(L[:,0:1]+L[:,1:]*1j)
                tR_angle=np.angle(tR[:,0:1]+tR[:,1:]*1j)
                R_angle=np.angle(R[:,0:1]+R[:,1:]*1j)

                #print(tL_angle.shape,output['binaural_spectrogram'].shape)
                # ((32, 1, 256, 64), (32, 2, 256, 64))


                # IPD has positive or negative   0.75pi - (0.75pi) = 1.5 pi but it is -0.5pi;  pi - - pi= 2pi but should 0
                # -0.95 - 0.95= -1.9 --> 0.1   0.8 - -0.8=1.6  
                tIPD=tL_angle-tR_angle 
                tIPD[tIPD<-np.pi]=2*np.pi+tIPD[tIPD<-np.pi]
                tIPD[tIPD>np.pi]=tIPD[tIPD>np.pi]-2*np.pi

                IPD=L_angle-R_angle 
                IPD[IPD<-np.pi]=2*np.pi+IPD[IPD<-np.pi]
                IPD[IPD>np.pi]=IPD[IPD>np.pi]-2*np.pi

                phase_error=np.abs(IPD-tIPD)
                phase_error[phase_error>np.pi]=2*np.pi-phase_error[phase_error>np.pi]

                
                #print(tL.shape,tL_angle.shape,tIPD.shape,phase_error.shape)
                #((32, 2, 256, 64), (32, 1, 256, 64), (32, 1, 256, 64), (32, 1, 256, 64))

                logMagPhaseError=np.sum(logMagM*(phase_error),axis=(2,3))/np.sum(logMagM,axis=(2,3))
                #print("shape of weighted average bias",logMagBias.shape)
                logMagPhaseError=np.mean(logMagPhaseError)
                #print("shape of mean of weighted average bias",logMagBias.shape)
                magPhaseError=np.sum(magM*(phase_error),axis=(2,3))/np.sum(magM,axis=(2,3))
                magPhaseError=np.mean(magPhaseError)


                magPhaseErrors.append(magPhaseError) 
                logMagPhaseErrors.append(logMagPhaseError)

            else:
                break
    avg_loss = sum(magBiases)/len(magBiases)
    avg_logloss = sum(logMagBiases)/len(logMagBiases)

    avg_phase_error = sum(magPhaseErrors)/len(magPhaseErrors)
    avg_logphase_error = sum(logMagPhaseErrors)/len(logMagPhaseErrors)

    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss)
    print('bias log: %.3f' % avg_logloss)
    print('bias: %.3f' % avg_loss)
    print('phase log: %.3f' % avg_logphase_error)
    print('phase: %.3f' % avg_phase_error)

    return avg_loss
 

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")


#create validation set data loader if validation_on option is set
if opt.validation_on:
    #temperally set to val to load val data
    opt.mode = 'val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#validation clips = %d' % dataset_size_val)
    opt.mode = 'train' #set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# network builders
builder = ModelBuilder()
net_visual = builder.build_visual(weights=opt.weights_visual)
net_audio = builder.build_audio(
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_audio,relu=True,sigmoid=False,batch_norm=True)
nets = (net_visual, net_audio)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)



# set up loss function
loss_criterion = torch.nn.MSELoss()
if(len(opt.gpu_ids) > 0):
    loss_criterion.cuda(opt.gpu_ids[0])
print("begin evaluation")
model.eval()
opt.mode = 'val'
display_otherMetric(model, writer, dataset_val, opt)
val_err = display_val(model, loss_criterion, writer, dataset_val, opt)


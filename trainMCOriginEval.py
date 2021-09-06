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
from models.modelRatioMCOrigin2in_audioVisual import AudioVisualModel
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def create_optimizer(nets, opt):
    (net_visual, net_audio) = nets
    param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_audio.parameters(), 'lr': opt.lr_audio}]
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

#used to display validation loss
def display_val(model, loss_criterion, writer, index, dataset_val, opt):
    batch_loss = []
    batch_mse_loss = []
    batch_bm_loss = []
    batch_phase_loss = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                mse_loss = loss_criterion(output['predicted_spectrogram'], output['audio_gt'])
                bm_loss = loss_criterion(output['BM_pred'], output['BM_gt'])
                loss=bm_loss+mse_loss # mse_loss+
                batch_loss.append(loss.item())
                batch_mse_loss.append(mse_loss.item())
                batch_bm_loss.append(bm_loss.item())
            else:
                break
    avg_loss = sum(batch_loss)/len(batch_loss)
    avg_mse_loss = sum(batch_mse_loss)/len(batch_mse_loss)
    avg_bm_loss = sum(batch_bm_loss)/len(batch_bm_loss)
    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss, index)
        writer.add_scalar('data/mse_loss', avg_mse_loss, index)
        writer.add_scalar('data/bm__loss', avg_bm_loss, index)
    print('val loss: %.3f' % avg_loss)
    print('val mse loss: %.3f' % avg_mse_loss)
    print('val bm  loss: %.3f' % avg_bm_loss)
    return avg_loss


def display_otherMetric(model, writer, dataset_val, opt):
    eps=1e-8
    eps2=1e-4
    magBiases = []
    logMagBiases = []
    magBiases2 = []
    logMagBiases2 = []
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
                magHatD=(output['BM_pred'])
                magGtD=(output['BM_gt'])
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
                # bias done  phase to go!
                # reuse the logMag and 


            else:
                break
    avg_loss = sum(magBiases)/len(magBiases)
    avg_logloss = sum(logMagBiases)/len(logMagBiases)

    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss)
    print('bias log: %.3f' % avg_logloss)
    print('bias: %.3f' % avg_loss)

    return avg_loss
 

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

#construct data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training clips = %d' % dataset_size)

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

# set up optimizer
optimizer = create_optimizer(nets, opt)

# set up loss function
loss_criterion = torch.nn.MSELoss()
if(len(opt.gpu_ids) > 0):
    loss_criterion.cuda(opt.gpu_ids[0])

# initialization
total_steps = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_loss = []
batch_mse_loss = []
batch_bm_loss = []
batch_phase_loss = []
best_err = float("inf")

for epoch in range(1, opt.niter+1):
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        if(opt.measure_time):
                iter_start_time = time.time()
        for i, data in enumerate(dataset):
                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_loaded_time = time.time()

                total_steps += opt.batchSize

                # forward pass
                model.zero_grad()
                output = model.forward(data)

                # compute loss
                mse_loss = loss_criterion(output['predicted_spectrogram'], output['audio_gt'])
                bm_loss = loss_criterion(output['BM_pred'], output['BM_gt'])
                loss=bm_loss+mse_loss # mse_loss
                batch_loss.append(loss.item())
                batch_mse_loss.append(mse_loss.item())
                batch_bm_loss.append(bm_loss.item())

                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_forwarded_time = time.time()

                # update optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if(opt.measure_time):
                        iter_model_backwarded_time = time.time()
                        data_loading_time.append(iter_data_loaded_time - iter_start_time)
                        model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                        model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                if(total_steps // opt.batchSize % opt.display_freq == 0):
                        print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        avg_loss = sum(batch_loss) / len(batch_loss)
                        print('Average loss: %.3f' % (avg_loss))
                        batch_loss = []
                        avg_loss = sum(batch_mse_loss) / len(batch_mse_loss)
                        print('mse     loss: %.3f' % (avg_loss))
                        batch_mse_loss = []
                        avg_loss = sum(batch_bm_loss) / len(batch_bm_loss)
                        print('Bratio  loss: %.3f' % (avg_loss))
                        batch_bm_loss = []
                        if opt.tensorboard:
                            writer.add_scalar('data/loss', avg_loss, total_steps)
                        if(opt.measure_time):
                                print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                                print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                                print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                                data_loading_time = []
                                model_forward_time = []
                                model_backward_time = []
                        # print('end of display \n')

                if(total_steps // opt.batchSize % opt.save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                        torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_latest.pth'))
                        torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_latest.pth'))

        if(epoch % opt.validation_freq == (opt.validation_freq-1) and opt.validation_on):
                print("begin evaluation")
                model.eval()
                opt.mode = 'val'
                print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                val_err = display_val(model, loss_criterion, writer, total_steps, dataset_val, opt)
                print('end of display \n')
                model.train()
                opt.mode = 'train'
                #save the model that achieves the smallest validation error
                if val_err < best_err:
                    best_err = val_err
                    print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                    torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_best.pth'))
                    torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_best.pth'))

                    print("\r\n eval bias in the best model")
                    display_otherMetric(model, writer, dataset_val, opt)

        if(opt.measure_time):
                iter_start_time = time.time()

        if(epoch % opt.save_epoch_freq == 0):
                print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
                torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_visual.pth'))
                torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_audio.pth'))

        #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
        if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
            decrease_learning_rate(optimizer, opt.decay_factor)
            print('decreased learning rate by ', opt.decay_factor)

#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import librosa
import numpy as np
from PIL import Image
import subprocess
from options.test_options import TestOptions
import torchvision.transforms as transforms
import torch
from models.customBM_models import ModelBuilder
from models.modelRatioPMOrigin2in_audioVisual import AudioVisualModel
from data.dMonoRatio_audioVisual_dataset import generate_spectrogram
# for PaperMono
def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return rms / desired_rms, samples, rms

def audio_normalize_rms(samples, rms, desired_rms = 0.1, eps = 1e-4):
  samples = samples * (desired_rms / rms)
  return samples

def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")

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
    model.eval()
 

    print("\r\n\r\n start_vid num_vid:",opt.start_vid,opt.num_vid)
    for num in range(opt.start_vid,opt.start_vid+opt.num_vid):
        vid='%06d'%num
        # the path is only the dictionary now
        full_path=opt.input_audio_path+"/"+vid+".wav"
        video_full_path=opt.video_frame_path+"/"+vid+".mp4"
        mono_path=mono_path=full_path.replace('binaural_audios/','mono/mono') # be sure don't add / at the end of opt.input_audio_path
        print("\r\nPath:\r\n%s\r\n%s\r\n%s",full_path,mono_path,video_full_path)
        # continue
        #load the audio to perform separation
        audio, audio_rate = librosa.load(full_path, sr=opt.audio_sampling_rate, mono=False)
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]
        mono_audio, audio_rate = librosa.load(mono_path, sr=opt.audio_sampling_rate, mono=False)
        

        #define the transformation to perform on visual frames
        vision_transform_list = [transforms.Resize((224,448)), transforms.ToTensor()]
        vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        vision_transform = transforms.Compose(vision_transform_list)

        #perform spatialization over the whole audio using a sliding window approach
        overlap_count = np.zeros((audio.shape)) #count the number of times a data point is calculated
        binaural_audio = np.zeros((audio.shape))

        #perform spatialization over the whole spectrogram in a siliding-window fashion
        sliding_window_start = 0
        data = {}
        samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
        while sliding_window_start + samples_per_window < audio.shape[-1]:
            sliding_window_end = sliding_window_start + samples_per_window
            normalizer, audio_segment, rms = audio_normalize(audio[:,sliding_window_start:sliding_window_end])
            audio_segment_channel1 = audio_segment[0,:]
            audio_segment_channel2 = audio_segment[1,:]

            if opt.M=='L':
              audio_segment_mix = audio_normalize_rms(audio_channel1[sliding_window_start:sliding_window_end],rms)#audio_segment_channel1 + audio_segment_channel2
            elif opt.M=='R':
              audio_segment_mix = audio_normalize_rms(audio_channel2[sliding_window_start:sliding_window_end],rms)#audio_segment_channel1 + audio_segment_channel2
            else:
              audio_segment_mix = audio_normalize_rms(mono_audio[sliding_window_start:sliding_window_end],rms)#audio_segment_channel1 + audio_segment_channel2

            data['audio_input_spec'] = torch.FloatTensor(generate_spectrogram(audio_segment_mix)).unsqueeze(0) #unsqueeze to add a batch dimension
            data['audio_target_spec'] = torch.FloatTensor([1]).unsqueeze(0).unsqueeze(0).unsqueeze(0) # only for gt, here we don't use it. torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
            data['audio_ratio_target_spec'] = torch.FloatTensor([[1,1],[1,1]]).unsqueeze(2).unsqueeze(0) # only for gt, here we don't use it. torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
            
            #get the frame index for current window
            frame_index = int(round((((sliding_window_start + samples_per_window / 2.0) / audio.shape[-1]) * opt.input_audio_length + 0.05) * 10 ))
            image = Image.open(os.path.join(video_full_path, str(frame_index).zfill(6) + '.png')).convert('RGB')
            #image = image.transpose(Image.FLIP_LEFT_RIGHT)
            frame = vision_transform(image).unsqueeze(0) #unsqueeze to add a batch dimension
            data['frame'] = frame

            output = model.forward(data)
            predicted_spectrogram = output['predicted_spectrogram'][0,:,:,:].data[:].cpu().numpy()

            #ISTFT to convert back to audio
            reconstructed_stft_diff = predicted_spectrogram[0,:,:] + (1j * predicted_spectrogram[1,:,:])
            reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=160, win_length=400, center=True, length=samples_per_window)
            reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) # M(1+B)= M + MB
            reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) 
            reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

            binaural_audio[:,sliding_window_start:sliding_window_end] = binaural_audio[:,sliding_window_start:sliding_window_end] + reconstructed_binaural
            overlap_count[:,sliding_window_start:sliding_window_end] = overlap_count[:,sliding_window_start:sliding_window_end] + 1
            sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

        #deal with the last segment
        normalizer, audio_segment, rms = audio_normalize(audio[:,-samples_per_window:])
        audio_segment_channel1 = audio_segment[0,:]
        audio_segment_channel2 = audio_segment[1,:]


        if opt.M=='L':
          audio_segment_mix = audio_normalize_rms(audio_channel1[-samples_per_window:],rms)#audio_segment_channel1 + audio_segment_channel2
          print("Use L as Input")
        elif opt.M=='R':
          audio_segment_mix = audio_normalize_rms(audio_channel2[-samples_per_window:],rms)#audio_segment_channel1 + audio_segment_channel2
          print("Use R as Input")
        else:
          audio_segment_mix = audio_normalize_rms(mono_audio[-samples_per_window:],rms)#audio_segment_channel1 + audio_segment_channel2
          print("Use M as input")
        data['audio_input_spec'] = torch.FloatTensor(generate_spectrogram(audio_segment_mix)).unsqueeze(0) #unsqueeze to add a batch dimension
        data['audio_target_spec'] = torch.FloatTensor([1]).unsqueeze(0).unsqueeze(0).unsqueeze(0) # only for gt, here we don't use it. torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
        data['audio_ratio_target_spec'] = torch.FloatTensor([[1,1],[1,1]]).unsqueeze(2).unsqueeze(0) # only for gt, here we don't use it. torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
        
        #get the frame index for last window
        frame_index = int(round(((opt.input_audio_length - opt.audio_length / 2.0) + 0.05) * 10))
        image = Image.open(os.path.join(video_full_path, str(frame_index).zfill(6) + '.png')).convert('RGB')
        #image = image.transpose(Image.FLIP_LEFT_RIGHT)
        frame = vision_transform(image).unsqueeze(0) #unsqueeze to add a batch dimension
        data['frame'] = frame
        output = model.forward(data)
        predicted_spectrogram = output['predicted_spectrogram'][0,:,:,:].data[:].cpu().numpy()
        #ISTFT to convert back to audio
        reconstructed_stft_diff = predicted_spectrogram[0,:,:] + (1j * predicted_spectrogram[1,:,:])
        reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=160, win_length=400, center=True, length=samples_per_window)
        reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) # M(1+B)= M + MB
        reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) 
        reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

        #add the spatialized audio to reconstructed_binaural
        binaural_audio[:,-samples_per_window:] = binaural_audio[:,-samples_per_window:] + reconstructed_binaural
        overlap_count[:,-samples_per_window:] = overlap_count[:,-samples_per_window:] + 1

        #divide aggregated predicted audio by their corresponding counts
        predicted_binaural_audio = np.divide(binaural_audio, overlap_count)

        #check output directory
        if not os.path.isdir(opt.output_dir_root):
            os.mkdir(opt.output_dir_root)

        mixed_mono = mono_audio
        fileName=full_path.split('/')[-1]
        print("write %s%s"%(opt.prefix,fileName))
        librosa.output.write_wav(os.path.join(opt.output_dir_root, '%s%s'%(opt.prefix,fileName)), predicted_binaural_audio, opt.audio_sampling_rate)
        # librosa.output.write_wav(os.path.join(opt.output_dir_root, 'ffmpeg_mono%s'%(fileName)), mixed_mono, opt.audio_sampling_rate)
        # librosa.output.write_wav(os.path.join(opt.output_dir_root, 'gt%s'%(fileName)), audio, opt.audio_sampling_rate)

if __name__ == '__main__':
    main()

#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        
        self.parser.add_argument('--input_audio_path', required=True, help='path to the input audio file')
        self.parser.add_argument('--video_frame_path', required=True, help='path to the input video frames')
        self.parser.add_argument('--output_dir_root', type=str, default='test_output', help='path to the output files')
        self.parser.add_argument('--input_audio_length', type=float, default=10, help='length of the testing video/audio')
        self.parser.add_argument('--hop_size', default=0.05, type=float, help='the hop length to perform audio spatialization in a sliding window approach')
        
        #model arguments
        self.parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream")
        self.parser.add_argument('--weights_audio', type=str, default='', help="weights for audio stream")
        self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
        self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
        self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")
        
        #my arguments
        self.parser.add_argument('--prefix', type=str, default="", help="prefix for output files")
        self.parser.add_argument('--ids', type=str, default="", help="multifiles generation")
        self.parser.add_argument('--start_vid', type=int, default=0, help="multifile generation from vid")
        self.parser.add_argument('--num_vid', type=int, default=0, help="number of multifile generation from vid")
        
        self.mode = "test"
        self.isTrain = False
        
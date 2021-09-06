#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.model == 'audioVisual':
        from data.audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
        print("have used data:",opt.model)
    elif opt.model == 'audioVisualMono':
        print("have used data:",opt.model)    
        from data.dMono_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    elif opt.model == 'audioVisualMonoRatio': 
        print("have used data:",opt.model)   
        from data.dMonoRatio_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    elif opt.model == 'audioVisualMonoSfRatio':
        print("have used data:",opt.model)    
        from data.dMonoSfRatio_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
        print("have used data:",opt.model)
    elif opt.model == 'audioVisualMono2p1dRatio':    
        from data.dMono2p1dRatio_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
        print("have used data:",opt.model)
        
        
        
    elif opt.model == 'audioVisualMonoRatioFT':    
        from data.dMonoRatioFT_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    elif opt.model == 'audioVisualMonoRatioMag':    
        from data.dMonoRatioMag_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    elif opt.model == 'audioVisualMonoRatioMagPool':    
        from data.dMonoRatioMagPool_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    elif opt.model == 'audioVisualMonoUniRatioMag':    
        from data.dMonoUniRatio_audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
            
    
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        shuffle=True if opt.mode=='train' else False
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuffle,
            num_workers=int(opt.nThreads))
        print("mode:%s, shuffle:"%opt.mode,shuffle)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

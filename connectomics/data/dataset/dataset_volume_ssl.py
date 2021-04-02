from __future__ import print_function, division
from typing import Optional, List
import numpy as np
import random
from ..dataset.dataset_volume import VolumeDataset
import torch
import torch.utils.data
from ..augmentation import Compose
from ..utils import *

class VolumeDatasetUDA(VolumeDataset):

    def __init__(self,shared_kwargs, volume,iter_num):
        super(VolumeDatasetUDA, self).__init__(volume = volume, iter_num = iter_num, **shared_kwargs)

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32

        vol_size = self.sample_volume_size
        if self.mode == 'train':
            sample = self._get_uda_samples(vol_size)
            return self._process_targets(sample)

    def _process_targets(self, sample):
        pos, out_volume_aug_one, out_volume_aug_two = sample

        if self.do_2d:
            out_volume_aug_one = np.squeeze(out_volume_aug_one)
            out_volume_aug_two = np.squeeze(out_volume_aug_two)

        out_volume_aug_one = np.expand_dims(out_volume_aug_one, 0)
        out_volume_aug_one = normalize_image(out_volume_aug_one, self.data_mean, self.data_std)
        
        out_volume_aug_two = np.expand_dims(out_volume_aug_two, 0)
        out_volume_aug_two = normalize_image(out_volume_aug_two, self.data_mean, self.data_std)
        # output list
        
        return pos, out_volume_aug_one, out_volume_aug_two
    
    def _get_uda_samples(self, vol_size):
        """Rejection sampling to filter out samples without required number 
        of foreground pixels or valid ratio.
        """
        sample = self._random_sampling(vol_size)
        pos, out_volume, out_label, out_valid = sample
        if self.augmentor is not None:
            data = {'image': out_volume,
                    'label': out_label,
                    'valid_mask': out_valid}

            augmented = self.augmentor(data)
            out_volume_aug_one = augmented['image']

            augmented = self.augmentor(data)
            out_volume_aug_two = augmented['image']

        return pos, out_volume_aug_one, out_volume_aug_two
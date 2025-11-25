# This code is referenced from
# https://github.com/facebookresearch/astmt/
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# License: Attribution-NonCommercial 4.0 International
#
# MODIFIED to support a different data storage format for NYUDv2
# where data is pre-split into train/test folders and stored as .npy files.

import os
import sys
import fnmatch  # Import fnmatch to count files

# Unused imports from the original file are removed:
# import tarfile
# import cv2
# from PIL import Image
# import scipy.io as sio
# from six.moves import urllib

import numpy as np
import torch.utils.data as data
import torch


class NYUD_MT(data.Dataset):
    """
    MODIFIED NYUD dataset class for multi-task learning.
    This version is adapted to load data from a directory structure like:
    - root/
      - train/
        - image/
          - 0.npy
          - 1.npy
          ...
        - label/
        - depth/
        - normal/
      - test/
        ... (same structure as train)

    Includes semantic segmentation, depth prediction, and surface normals.
    The 'edge' task has been removed as per request.
    """

    def __init__(self,
                 p=None,
                 root=None,
                 split='train',  # Expects 'train' or 'test' (or other sub-folders)
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 ):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.retname = retname
        self.overfit = overfit
        self.p = p

        # Set the specific data path based on the split
        self.data_path = os.path.join(self.root, self.split)
        if not os.path.isdir(self.data_path):
            raise RuntimeError(f"Split data path not found: {self.data_path}")

        # The new loading mechanism enables/disables tasks
        self.do_semseg = do_semseg
        self.do_normals = do_normals
        self.do_depth = do_depth

        # Calculate data length by counting .npy files in the image directory
        # This replaces reading from a split file (e.g., train.txt)
        image_dir = os.path.join(self.data_path, 'image')
        self.data_len = len(fnmatch.filter(os.listdir(image_dir), '*.npy'))

        if self.data_len == 0:
            raise RuntimeError(f"No .npy files found in {image_dir}")

        # If overfitting, limit the number of samples
        if self.overfit:
            self.data_len = 64 # You can change this number for overfitting tests

        # Display stats
        print(f"Initializing dataloader for NYUD '{self.split}' set.")
        print(f"Number of dataset images: {self.data_len}")

        labelroot = './data/ssl_mapping/nyud_low/'
        if split == 'train':
            if self.p.ssl_type == 'onelabel':
                self.labels_weights = torch.load('{}onelabel.pth'.format(labelroot))['labels_weights'].float()
            elif self.p.ssl_type == 'randomlabels':
                self.labels_weights = torch.load('{}randomlabels.pth'.format(labelroot))['labels_weights'].float()


    def __getitem__(self, index):
        """
        Loads a single data sample by index.
        Paths are generated on-the-fly using the index.
        """
        sample = {}

        # --- Load Image ---
        # All data is expected to be in .npy format
        img_path = os.path.join(self.data_path, 'image', f'{index}.npy')
        # Load raw image to get original shape for meta info and asserts
        _img_raw = np.load(img_path).astype(np.float32) * 255.
        _img = _img_raw
        sample['image'] = _img

        # --- Load Semantic Segmentation (Label) ---
        if self.do_semseg:
            semseg_path = os.path.join(self.data_path, 'label', f'{index}.npy')
            _semseg = np.load(semseg_path).astype(np.int64)[:, :, np.newaxis] # Labels are usually long type
            _semseg = np.where(_semseg == -1, 255, _semseg)
            sample['semseg'] = _semseg

        # --- Load Surface Normals ---
        if self.do_normals:
            normal_path = os.path.join(self.data_path, 'normal', f'{index}.npy')
            _normals = np.load(normal_path).astype(np.float32)
            sample['normals'] = _normals

        # --- Load Depth ---
        if self.do_depth:
            depth_path = os.path.join(self.data_path, 'depth', f'{index}.npy')
            _depth = np.load(depth_path)
            if _depth.ndim == 2:
                _depth = np.expand_dims(_depth, axis=-1) # Add channel dim: (H, W, 1)
            sample['depth'] = _depth.astype(np.float32)

        # --- Load Metadata ---
        if self.retname:
            sample['meta'] = {'img_name': index,
                              'img_size': (_img.shape[0], _img.shape[1])}

        # --- Apply Transforms ---
        if self.transform is not None:
            sample = self.transform(sample)

        if self.split == 'train':
            image_index = sample['meta']['img_name']
            w = torch.tensor(self.labels_weights[image_index]) #.clone().float().cuda()
            tasks = self.p.TASKS.NAMES
            for t_idx, task in enumerate(tasks):
                if not w[t_idx] == 1:
                    # we should not know this label in training
                    sample[task] = torch.zeros_like(sample[task])
                    if 'mask_'+task in sample.keys():
                        sample['mask_'+task] = torch.zeros_like(sample['mask_'+task])

            sample['task_w'] = w

        return sample

    def __len__(self):
        return self.data_len

    def __str__(self):
        return f"NYUD_MT(split={self.split}, len={self.data_len})"

    # The original _load_* methods are no longer needed as their logic
    # is now integrated directly into __getitem__ for .npy files.
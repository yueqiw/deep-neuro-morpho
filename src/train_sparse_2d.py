
import torch
import torch.nn as nn
import sparseconvnet as scn
import numpy as np 
import pandas as pd
import argparse
import os
import shutil
import time
import datetime
import json
from sklearn.model_selection import train_test_split

import sys
from data_utils import *
from models import SparseResNet2D 
pd.set_option('precision', 3)

np_dtype = 'float32'
cpu_dtype = torch.FloatTensor
gpu_dtype = torch.cuda.FloatTensor

print(torch.__version__ + '\n')

use_gpu = torch.cuda.is_available()
print('GPU: ' + str(use_gpu))


def main():

    root_dir = '~/Dropbox/lib/deep_neuro_morpho/data'
    data_dir = 'png_mip_256_fit_2d'
    classes = np.arange(6)
    n_classes = len(classes)
    batch_size = 64


    metadata = pd.read_pickle('../data/rodent_3d_dendrites_br-ct-filter-3_all_mainclasses_use_filter.pkl')
    metadata = metadata[metadata['label1_id'].isin(classes)]
    neuron_ids = metadata['neuron_id'].values
    labels = metadata['label1_id'].values  # contain the same set of values as classes
    unique, counts = np.unique(labels, return_counts=True)

    transform_train = None

    transform_test = None

    train_ids, test_ids, train_y, test_y = \
        train_test_split(neuron_ids, labels, test_size=0.15, random_state=42, stratify=labels)

    train_ids, val_ids, train_y, val_y = \
        train_test_split(train_ids, train_y, test_size=0.15, random_state=42, stratify=train_y)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        NeuroMorpho(root_dir, data_dir, train_ids, train_y, img_size=256,
                         transform=transform_train, rgb=False),
        collate_fn=SparseMerge(), 
        batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        NeuroMorpho(root_dir, data_dir, val_ids, val_y, img_size=256,
                         transform=transform_test, rgb=False),
        collate_fn=SparseMerge(), 
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        NeuroMorpho(root_dir, data_dir, test_ids, test_y, img_size=256,
                         transform=transform_test, rgb=False),
        collate_fn=SparseMerge(), 
        batch_size=batch_size, shuffle=True, **kwargs)

    model = SparseResNet2D(n_classes)
    dataset = {'train': train_loader, 'val': val_loader}
    print('Input spatial size:', model.spatial_size)

    scn.ClassificationTrainValidate(
        model, dataset,
        {'n_epochs': 100,
        'initial_lr': 0.1,
        'lr_decay': 0.05,
        'weight_decay': 1e-4,
        'use_cuda': torch.cuda.is_available(),
        'check_point': False, })

    

if __name__ == '__main__':
    main()








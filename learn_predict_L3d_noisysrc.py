# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:47:26 2017

@author: subhayanmukherjee
"""

print(__doc__)

import os
import os.path
import numpy as np
from LNet import Lnet
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
from pathlib import Path
import cv2

def generate_data(source_files, labels, noise_lvl):
    examples_cnt = len(source_files)
    while 1:
        rand_idx = np.random.permutation(examples_cnt)
        for loop_idx in range(examples_cnt):
            path_in_str = source_files[rand_idx[loop_idx]]
            
            path_split = path_in_str[:-4]   # remove file extension
            testfile_dir = path_split + '_NoiseLevel_' + str(noise_lvl)
            noisysrc_loc = testfile_dir + '/noisy_source_' + str(noise_lvl) + '.png'
            
            im_noisysrc = cv2.imread(noisysrc_loc)
            im_noisysrc = im_noisysrc[:,:,0].astype(np.float32)
            im_noisysrc = np.expand_dims(im_noisysrc, axis=0)
            im_noisysrc = np.expand_dims(im_noisysrc, axis=-1)
            
            yield (im_noisysrc, labels[rand_idx[loop_idx]])

path_prefix = '/home/subhayanmukherjee/Documents/'

noise_lvl = 30           # AWGN sigma value



bm3d_cpu = True

if bm3d_cpu:
    path_qual = '_cpu'
else:
    path_qual = ''



if os.path.isfile(path_prefix + 'predict/train_L3d_assorted_mse' + path_qual + '_' + str(noise_lvl) + '.npy'):
    train_y = np.load(path_prefix + 'predict/train_L3d_assorted_mse' + path_qual + '_' + str(noise_lvl) + '.npy')
    train_y = np.expand_dims(train_y, axis=-1)
else:
    print('Training labels not available !')
    raise SystemExit(0)
if os.path.isfile(path_prefix + 'predict/test_L3d_assorted_mse' + path_qual + '_' + str(noise_lvl) + '.npy'):
    test_y = np.load(path_prefix + 'predict/test_L3d_assorted_mse' + path_qual + '_' + str(noise_lvl) + '.npy')
    test_y = np.expand_dims(test_y, axis=-1)
else:
    print('Test labels not available !')
    raise SystemExit(0)
if os.path.isfile(path_prefix + 'predict/train_L3d_filelist_assorted' + path_qual + '.npy'):
    train_filelist = np.load(path_prefix + 'predict/train_L3d_filelist_assorted' + path_qual + '.npy')
else:
    print('Training file list not available !')
    raise SystemExit(0)
if os.path.isfile(path_prefix + 'predict/test_L3d_filelist_assorted' + path_qual + '.npy'):
    test_filelist = np.load(path_prefix + 'predict/test_L3d_filelist_assorted' + path_qual + '.npy')
else:
    print('Test file list not available !')
    raise SystemExit(0)

#raise SystemExit(0)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_recons = 1

model = Lnet(None, None, num_recons, optimizer='adam')

# model.load_weights(path_prefix + 'results/weights_mse_' + str(noise_lvl) + '/weights.197-0.1957.hdf5')

tensorboard = TensorBoard(log_dir=path_prefix + 'results/logs_mse' + path_qual + '_' + str(noise_lvl))
checkpoint = ModelCheckpoint(path_prefix + 'results/weights_mse' + path_qual + '_' + str(noise_lvl) + '/weights.{epoch:02d}-{val_loss:.4f}.hdf5', save_weights_only=True)

model.fit_generator(initial_epoch=0, epochs=500, generator=generate_data(train_filelist, train_y, noise_lvl), steps_per_epoch=len(train_filelist), validation_data=generate_data(test_filelist, test_y, noise_lvl), validation_steps=len(test_filelist), callbacks=[tensorboard,checkpoint])

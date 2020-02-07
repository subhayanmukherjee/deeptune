# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:47:26 2017

@author: subhayanmukherjee
"""

print(__doc__)

from time import time
import numpy as np
from pathlib import Path
import cv2
from sklearn.metrics import mean_squared_error
import os
from subprocess import call
from ssim import ssim
from LNet import Lnet

def generate_data(source_files, labels, noise_lvl):
    examples_cnt = len(source_files)
    while 1:
        for loop_idx in range(examples_cnt):
            path_in_str = source_files[loop_idx]
            
            path_split = path_in_str[:-4]   # remove file extension
            testfile_dir = path_split + '_NoiseLevel_' + str(noise_lvl)
            noisysrc_loc = testfile_dir + '/noisy_source_' + str(noise_lvl) + '.png'
            
            im_noisysrc = cv2.imread(noisysrc_loc)
            im_noisysrc = im_noisysrc[:,:,0].astype(np.float32)
            im_noisysrc = np.expand_dims(im_noisysrc, axis=0)
            im_noisysrc = np.expand_dims(im_noisysrc, axis=-1)
            
            yield (im_noisysrc, labels[loop_idx])

path_prefix = '/home/subhayanmukherjee/Documents/'

bm3dobj_loc  = path_prefix + 'bm3d/bm3d'
# bm3dobj_loc  = path_prefix + 'bm3d-master/build/bm3d'

noise_lvl = 30      # AWGN sigma value

max_L = 3.0
min_L = 1.0



bm3d_cpu = False

if bm3d_cpu:
    path_qual = '_cpu'
else:
    path_qual = ''



if os.path.isfile(path_prefix + 'predict/test_L3d_assorted_mse' + path_qual + '_' + str(noise_lvl) + '.npy'):
    test_y = np.load(path_prefix + 'predict/test_L3d_assorted_mse' + path_qual + '_' + str(noise_lvl) + '.npy')
    test_y = np.expand_dims(test_y, axis=-1)
else:
    print('Test labels not available !')
    raise SystemExit(0)
if os.path.isfile(path_prefix + 'predict/test_L3d_filelist_assorted' + path_qual + '.npy'):
    test_filelist = np.load(path_prefix + 'predict/test_L3d_filelist_assorted' + path_qual + '.npy')
else:
    print('Test file list not available !')
    raise SystemExit(0)

pred_count = len(test_filelist)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
num_recons = 1

if True:
    model = Lnet(None, None, num_recons, optimizer='adam')
    model.load_weights(path_prefix + 'results/weights_mse' + path_qual + '_' + str(noise_lvl) + '/weights.99-0.0159.hdf5')
    pred_L3d = model.predict_generator(generator=generate_data(test_filelist, test_y, noise_lvl), steps=pred_count)
    np.save(path_prefix + 'predict/pred_L3d_noisysrc' + path_qual + '_' + str(noise_lvl) + '.npy', pred_L3d)
    raise SystemExit(0)

pred_L3d = np.load(path_prefix + 'predict/pred_L3d_noisysrc' + path_qual + '_' + str(noise_lvl) + '.npy')

total_time = time()

total_mse = 0
total_psnr = 0
total_ssim = 0


if bm3d_cpu:
    bm3dobj_loc  = path_prefix + 'bm3d-master/build/bm3d'
else:
    bm3dobj_loc  = path_prefix + 'bm3d/bm3d'


pred_idx = 0
for path in test_filelist:
    # because path is object not string
    path_in_str = str(path)
    
    path_split = path_in_str[:-4]   # remove file extension
    testfile_dir = path_split + '_NoiseLevel_' + str(noise_lvl)
    noisysrc_loc = testfile_dir + '/noisy_source_' + str(noise_lvl) + '.png'
    
    name_split = path_in_str.split('/')
    noisy_name = name_split[-1]
    print(noisy_name)
    # denoised_loc = path_prefix + 'cuda-bm3d/outputs-noisysrc_' + str(noise_lvl) + '/denoised_' + noisy_name
    denoised_loc = path_prefix + 'predict/outputs-noisysrc' + path_qual + '_' + str(noise_lvl) + '/denoised_' + noisy_name
    
    L3dval = pred_L3d[pred_idx]
    
    L3dval = np.round(np.squeeze(L3dval), decimals=4)
    # if L3dval > max_L:
    #     L3dval = 2.0
    # L3dval = np.minimum(L3dval, max_L)
    # L3dval = np.maximum(L3dval, min_L)
    
    # L3dval = 2.7      # if you want to test BM3D performance with its default value of L3d
    
    # print(bm3dobj_loc + ' ' + noisysrc_loc + ' ' + denoised_loc + ' ' + str(noise_lvl) + ' gray twostep quiet ' + str(L3dval))
    if bm3d_cpu:
        call([bm3dobj_loc, noisysrc_loc, str(noise_lvl), denoised_loc, str(L3dval)])
    else:
        call([bm3dobj_loc, noisysrc_loc, denoised_loc, str(noise_lvl), 'gray', 'twostep', 'quiet', str(L3dval)])
    
    rec = cv2.imread(denoised_loc)
    rec = rec[:,:,0].astype(np.float32)
    
    im_clean = cv2.imread(path_in_str)
    im_clean = im_clean[:,:,0].astype(np.float32)
    
    this_mse = mean_squared_error(im_clean,rec)
    total_mse += this_mse
    
    this_psnr = 20. * np.log10(255.) - 10. * np.log10(this_mse)
    total_psnr += this_psnr
    
    this_ssim = ssim(im_clean,rec)
    total_ssim += this_ssim
    
    pred_idx += 1
    
    print('\nCurrent image predicted L3d value is %.4f' % L3dval)
    print('Current image PSNR = %.2f' % this_psnr)
    print('Current image SSIM = %.2f' % this_ssim)
    print('Current image MSE = %.2f\n' % this_mse)

print('\nAverage PSNR = %.2f' % (total_psnr/pred_count))
print('Average SSIM = %.4f' % (total_ssim/pred_count))
print('Average MSE = %.2f\n' % (total_mse/pred_count))

print('Total time taken for processing = %.2fs.\n' % (time()-total_time))

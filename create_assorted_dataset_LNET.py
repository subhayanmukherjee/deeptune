# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:47:26 2017

@author: subhayanmukherjee
"""

print(__doc__)

from time import time
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import mean_squared_error
from ssim import ssim
from itertools import chain
import matplotlib.pyplot as plt
import os
from subprocess import call

train = False
bm3d_cpu = True

def add_gaussian_noise(image_in, noise_sigma):
#    temp_image = np.float(np.copy(image_in))
    temp_image = image_in

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
        
    return noisy_image

path_prefix = '/home/subhayanmukherjee/Documents/'

noise_lvl = 30      # AWGN sigma value

max_L = 3.0
min_L = 1.0
L_stp = 0.125
num_recons = int( (max_L - min_L) / L_stp + 1 )

if train:
    data_cat = 'train'
    dataset_length = (1381+3254)
else:
    data_cat = 'test'
    dataset_length = 68

if bm3d_cpu:
    path_qual = '_cpu'
else:
    path_qual = ''

dataset_filelist_loc = path_prefix + 'predict/' + data_cat + '_L3d_filelist_assorted' + path_qual + '.npy'
dataset_mse_loc = path_prefix + 'predict/' + data_cat + '_L3d_assorted_mse' + path_qual + '_' + str(noise_lvl) + '.npy'
dataset_ssim_loc = path_prefix + 'predict/' + data_cat + '_L3d_assorted_ssim' + path_qual + '_' + str(noise_lvl) + '.npy'
if not os.path.isfile(dataset_mse_loc):
    best_L3d_mse = np.zeros(dataset_length, dtype=np.float)
else:
    best_L3d_mse = np.load(dataset_mse_loc)
if not os.path.isfile(dataset_ssim_loc):
    best_L3d_ssim = np.zeros(dataset_length, dtype=np.float)
else:
    best_L3d_ssim = np.load(dataset_ssim_loc)

total_time = time()
img_cnt = 0

if bm3d_cpu:
    bm3dobj_loc  = path_prefix + 'bm3d-master/build/bm3d'
else:
    bm3dobj_loc  = path_prefix + 'bm3d/bm3d'

if train:
    pathlist_jpg = Path(path_prefix + 'datasets/coco10k').glob('**/*.jpg')
    pathlist_tif = Path(path_prefix + 'datasets/mcgill').glob('**/*.tif')
    pathlist = chain(pathlist_jpg, pathlist_tif)
else:
    pathlist = Path(path_prefix + 'datasets/BSD68').glob('*.png')

dataset_filelist = []
for path in pathlist:
    # if best_L3d_mse[img_cnt] > 0:
    #     img_cnt += 1
    #     continue
    
    # because path is object not string
    path_in_str = str(path)
    
    dataset_filelist.append(path_in_str)
    
    path_split = path_in_str[:-4]   # remove file extension
    testfile_dir = path_split + '_NoiseLevel_' + str(noise_lvl)
    
    if not os.path.exists(testfile_dir):
       os.makedirs(testfile_dir)
    
    # Create noisy source
    noisysrc_loc = testfile_dir + '/noisy_source_' + str(noise_lvl) + '.png'
    if not os.path.isfile(noisysrc_loc):
        image = cv2.imread(path_in_str)
        gray_image = add_gaussian_noise(image[:,:,0].astype(np.float), noise_lvl)
        # make into proper image data
        gray_image = gray_image.round().clip(0, 255)
        plt.imsave(noisysrc_loc, gray_image, cmap='gray', vmin=0, vmax=255)
        print('Created noisy image ' + noisysrc_loc)
    
    im_clean = cv2.imread(path_in_str)
    im_clean = im_clean[:,:,0].astype(np.float32)
    
    min_mse = np.inf
    max_ssim = np.NINF
    for recon_idx in range (0, num_recons):
        L3dval = min_L + recon_idx*L_stp
        denoised_loc = testfile_dir + '/denoised' + path_qual + '_L3d_' + str(L3dval) + '.png'
        
        if os.path.exists(denoised_loc):
            print('Loading existing L3d_value ' + str(L3dval) + ' reconstruction for file ' + str(path_in_str))
        else:
            # Syntax: ./bm3d NoisyImage.png DenoisedImage.png sigma [color [twostep [quiet]]]
            # print(bm3dobj_loc + ' ' + noisysrc_loc + ' ' + denoised_loc + ' ' + str(noise_lvl) + ' gray twostep quiet ' + str(L3dval))
            if bm3d_cpu:
                call([bm3dobj_loc, noisysrc_loc, str(noise_lvl), denoised_loc, str(L3dval)])
            else:
                call([bm3dobj_loc, noisysrc_loc, denoised_loc, str(noise_lvl), 'gray', 'twostep', 'quiet', str(L3dval)])
        
        rec = cv2.imread(denoised_loc)
        rec = rec[:,:,0].astype(np.float32)
        
        this_mse = mean_squared_error(im_clean,rec)
        if this_mse < min_mse:
            min_mse = this_mse
            best_L3d_mse[img_cnt] = L3dval
        this_ssim = ssim(im_clean,rec)
        if this_ssim > max_ssim:
            max_ssim = this_ssim
            best_L3d_ssim[img_cnt] = L3dval
    
    img_cnt += 1
    percent = img_cnt/float(dataset_length) * 100.0
    print('Completed: %.2f percent. Image #%d of %d processed successfully...' % (percent, img_cnt, dataset_length))
    
    if img_cnt % 15 == 0:
        np.save(dataset_filelist_loc, dataset_filelist)
        np.save(dataset_mse_loc, best_L3d_mse)
        np.save(dataset_ssim_loc, best_L3d_ssim)
    
#    if img_cnt > 50:
#        break

np.save(dataset_filelist_loc, dataset_filelist)
np.save(dataset_mse_loc, best_L3d_mse)
np.save(dataset_ssim_loc, best_L3d_ssim)

print('\nTotal time taken for building and saving dataset = %.2fs.' % (time()-total_time))

## Image-based parameter-tuning of a noise filter using Deep Learning

This project explores a new direction in explainable Artificial Intelligence (AI). Most state-of-art deep-learning based noise removal algorithms involve some form of neural networks trained end-to-end (input: noisy image, output: clean image). Such methods are inexplicable "black-boxes" since we do not have a clear understanding of their inner workings and how change in input data distribution can affect their denoising performance. In contrast, this project proposes and validates a CNN-based method that tunes the parameter of a computer vision based (fully explainable) denoising algorithm based only on the noisy input image. Thus, we get the best of both worlds: explain-ability of vision and learning-based regression.

Please cite the below [paper](https://doi.org/10.1007/978-3-030-27202-9_10) if you use the code in its original or modified form:

*S. Mukherjee, N. K. Kottayil, X. Sun, and I. Cheng, “CNN-Based Real-Time Parameter Tuning for Optimizing Denoising Filter Performance,” in F. Karray, A. Campilho, A. Yu (eds) Image Analysis and Recognition, ICIAR 2019, Lecture Notes in Computer Science, Springer International Publishing, vol 11662, pp. 112–125.*

## Guidelines

1. Build the training dataset by running the [dataset script](https://github.com/subhayanmukherjee/deeptune/blob/master/create_assorted_dataset_LNET.py) create_assorted_dataset_LNET.py, setting *train = True* for the desired *noise_lvl* inside the script. The script is written in a way that allows you to source images files with two different extensions from two folders while building the training dataset. This is similar to what we do in our [paper](https://doi.org/10.1007/978-3-030-27202-9_10).
2. To use the [GPU implementation of BM3D (BM3D-GPU)](https://github.com/DawyD/bm3d-gpu), as in our [paper](https://doi.org/10.1007/978-3-030-27202-9_10), compile the required binaries. Then, set *bm3d_cpu = False* inside the [dataset script](https://github.com/subhayanmukherjee/deeptune/blob/master/create_assorted_dataset_LNET.py) and set the *bm3dobj_loc* path to your compiled BM3D-GPU binary path.
3. The [model script](https://github.com/subhayanmukherjee/deeptune/blob/master/LNet.py) LNet.py has the implementation of the CNN we use to predict the &#955;<sub>3D</sub> parameter in our [paper](https://doi.org/10.1007/978-3-030-27202-9_10).

"""Functions for image processing
"""
# MIT License
# 
# Copyright (c) 2022 Debayan Deb

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import math
import random
import numpy as np
from scipy import misc
import imageio
import cv2
from PIL import Image       
from matplotlib.colors import rgb_to_hsv
from skimage.util import view_as_blocks
from scipy.special import expit

# Calulate the shape for creating new array given (h,w)
def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape


def patch_loc(images, spoof_maps, patch_map):
    _h = 256
    _w = 256
    n = len(images)

    spoof_maps = expit(spoof_maps)
    # print(spoof_maps.shape)
    locations = []
    for i in range(n):
        locs = np.where(spoof_maps[i] >= 0.5)

        if locs[0].size == 0:
            locs = [7, 7]
        locations.append(np.column_stack(locs))

    # print(image.shape, locations.shape)


    images_new = np.ndarray((len(images), _h, _w, 3), dtype=images.dtype)
    LOCIDX =  [random.randrange(0, len(locations[i])) for i in range(n)]
    offsetx = [random.randrange(64, _h, 2) for _ in range(n)]
    offsety = [random.randrange(64, _w, 2) for _ in range(n)]
    for i in range(len(images)):
        image_location = patch_map[locations[i][LOCIDX[i]][0], locations[i][LOCIDX[i]][1]]
        h = offsetx[i]
        w = offsety[i]

        if image_location[0] - offsetx[i]//2 >=  0 :
            startx = image_location[0] -  offsetx[i]//2
        else:
            startx = 0
        if image_location[1] - offsety[i]//2 >=  0 :
            starty = image_location[1] - offsety[i]//2
        else:
            starty = 0
        if image_location[0] + offsetx[i]//2 <=  256 :
            endx = image_location[0] +  offsetx[i]//2
        else:
            endx = 256
            startx = 256- offsetx[i]
        if image_location[1] + offsety[i]//2 <= 256 :
            endy = image_location[1] + offsety[i]//2
        else:
            endy = 256
            starty = 256-offsety[i]
        startx, starty, endx, endy = int(startx), int(starty), int(endx), int(endy)
        if (endx - startx) % 2 == 1:
            startx = startx + 1
        if (endy - starty) % 2 == 1:
            starty = starty + 1
        _im = images[i][startx:endx, starty:endy]

        h,w = _im.shape[0:2]
        images_new[i] = np.pad(_im, [(int((_h-h+1) / 2),), (int((_w-w+1) / 2),), (0,)], 
                'constant',
             constant_values=0)
    return images_new

def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    if tuple(size)[0] is not None:
        h, w = tuple(size)
        shape_new = get_new_shape(images, size)
        assert (_h>=h and _w>=w)

        images_new = np.ndarray(shape_new, dtype=images.dtype)

        y = np.random.randint(low=0, high=_h-h+1, size=(n))
        x = np.random.randint(low=0, high=_w-w+1, size=(n))

        for i in range(n):
            images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]
    else:
        h, w = tuple(size)
        images_new = np.ndarray(get_new_shape(images, (_h, _w)), dtype=images.dtype)

        for i in range(n):
            # h = np.random.randint(low=64, high=_h)
            # w = np.random.randint(low=64, high=_w)
            h = [random.randrange(32, _h, 2) for _ in range(1)][0]
            w = [random.randrange(32, _w, 2) for _ in range(1)][0]
            y = np.random.randint(low=0, high=_h-h+1, size=(n))
            x = np.random.randint(low=0, high=_w-w+1, size=(n))
            img = images[i, y[i]:y[i]+h, x[i]:x[i]+w]
            images_new[i] = np.pad(img, [(int((_h-h+1) / 2),), (int((_w-w+1) / 2),), (0,)], 
                'constant',
             constant_values=0)
    # else:
    #     h, w = tuple(size)
    #     images_new = np.ndarray(get_new_shape(images, (_h, _w)), dtype=images.dtype)

    #     h = np.array([random.randrange(64, _h, 2) for _ in range(n)])
    #     w = np.array([random.randrange(64, _w, 2) for _ in range(n)])
    #     y = np.random.randint(low=0, high=_h-h+1, size=(n))
    #     x = np.random.randint(low=0, high=_w-w+1, size=(n))
    #     mask = np.zeros(images.shape, images.dtype)
    #     for i in range(n):
    #         mask[i, y[i]:y[i]+h[i],x[i]:x[i]+w[i]] = images[i, y[i]:y[i]+h[i], x[i]:x[i]+w[i]]
    #         images_new[i] = mask[i]

    #         # img = images[i, y[i]:y[i]+h, x[i]:x[i]+w]
    #         # images_new[i] = np.pad(img, [(int((_h-h+1) / 2),), (int((_w-w+1) / 2),), (0,)], 
    #         #     'constant',
    #         #  constant_values=0)
    # else:
    #     h, w = tuple(size)
        

    #     h = np.array([random.randrange(64, _h, 2) for _ in range(1)])
    #     w = np.array([random.randrange(64, _w, 2) for _ in range(1)])
    #     y = np.random.randint(low=0, high=_h-h+1, size=(1))
    #     x = np.random.randint(low=0, high=_w-w+1, size=(1))
    #     images_new = np.ndarray(get_new_shape(images, (h[0], w[0])), dtype=images.dtype)
    #     # mask = np.zeros(images.shape, images.dtype)
    #     for i in range(n):
    #         # mask[i, y[i]:y[i]+h[i],x[i]:x[i]+w[i]] = images[i, y[i]:y[i]+h[i], x[i]:x[i]+w[i]]
    #         images_new[i] = images[i, y[0]:y[0]+h[0], x[0]:x[0]+w[0]]

    #         # img = images[i, y[i]:y[i]+h, x[i]:x[i]+w]
    #         # images_new[i] = np.pad(img, [(int((_h-h+1) / 2),), (int((_w-w+1) / 2),), (0,)], 
    #         #     'constant',
    #         #  constant_values=0)
    # else:
    #     size = [28, 28]
    #     h, w = tuple(size)
    #     shape_new = get_new_shape(images, size)
    #     assert (_h>=h and _w>=w)

    #     images_new = np.ndarray(shape_new, dtype=images.dtype)

    #     y = np.random.randint(low=0, high=_h-h+1, size=(n))
    #     x = np.random.randint(low=0, high=_w-w+1, size=(n))

    #     for i in range(n):
    #         images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]
    # print(images_new.shape)
    return images_new

def patches(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    patch_n = _h // h
    shape_new = (n, patch_n * patch_n, h, w, 3)

    images_new = np.ndarray(shape_new, dtype=np.float32)

    for i in range(n):
        images_new[i] = view_as_blocks(images[i],
         (h,w,3)).reshape((patch_n * patch_n, h, w, 3))
    return images_new

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

# def random_flip(images):
#     images_new = images
#     flips = np.random.rand(images_new[0].shape[0])>=0.5
    
#     for i in range(images_new[0].shape[0]):
#         if flips[i]:
#             idx = random.randint(0, 2)
#             images_new[idx][i] = np.fliplr(images[idx][i])

#     return images_new

def random_flip(images):
    images_new = images
    flips = np.random.rand(images_new.shape[0])>=0.5
    
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])
    return images_new

def flip(images):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = np.fliplr(images[i])

    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def padding(images, padding):
    n, _h, _w = images.shape[:3]
    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    else:
        pad_t, pad_b, pad_l, pad_r = tuple(padding)
       
    size_new = (_h + pad_t + pad_b, _w + pad_l + pad_r)
    shape_new = get_new_shape(images, size_new)
    images_new = np.zeros(shape_new, dtype=images.dtype)
    images_new[:, pad_t:pad_t+_h, pad_l:pad_l+_w] = images

    return images_new

# def standardize_images(images, standard):
#     if standard=='mean_scale':
#         mean = 127.5
#         std = 128.0
#     elif standard=='scale':
#         mean = 0.0
#         std = 255.0
#     global_images_new = images[0].astype(np.float32)
#     eye_images_new = images[1].astype(np.float32)
#     nose_images_new = images[2].astype(np.float32)
#     global_images_new = (global_images_new - mean) / std
#     eye_images_new = (eye_images_new - mean) / std
#     nose_images_new = (nose_images_new - mean) / std
#     return (global_images_new, eye_images_new, nose_images_new)

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new



def random_shift(images, max_ratio):
    n, _h, _w = images.shape[:3]
    pad_x = int(_w * max_ratio) + 1
    pad_y = int(_h * max_ratio) + 1
    images_temp = padding(images, (pad_y, pad_x))
    images_new = images.copy()

    shift_x = (_w * max_ratio * np.random.rand(n)).astype(np.int32)
    shift_y = (_h * max_ratio * np.random.rand(n)).astype(np.int32)

    for i in range(n):
        images_new[i] = images_temp[i, pad_y+shift_y[i]:pad_y+shift_y[i]+_h, 
                            pad_x+shift_x[i]:pad_x+shift_x[i]+_w]

    return images_new    
    
def random_rotate(images, max_degree):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    
    degree = max_degree * np.random.rand(n)

    for i in range(n):
        M = cv2.getRotationMatrix2D((_w/2, _h/2), int(degree[i]), 1)
        images = [cv2.warpAffine(img, M, (_w, _h)) for img in images]

    return images_new


def random_blur(images, blur_type, max_size):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    
    kernel_size = max_size * np.random.rand(n)
    
    for i in range(n):
        size = int(kernel_size[i])
        if size > 0:
            if blur_type == 'motion':
                kernel = np.zeros((size, size))
                kernel[int((size-1)/2), :] = np.ones(size)
                kernel = kernel / size
                img = cv2.filter2D(images[i], -1, kernel)
            elif blur_type == 'gaussian':
                size = size // 2 * 2 + 1
                img = cv2.GaussianBlur(images[i], (size,size), 0)
            else:
                raise ValueError('Unkown blur type: {}'.format(blur_type))
            images_new[i] = img

    return images_new
    
def random_noise(images, stddev, min_=-1.0, max_=1.0):

    noises = np.random.normal(0.0, stddev, images.shape)
    images_new = np.maximum(min_, np.minimum(max_, images + noises))
        
    return images_new

def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    ratios = min_ratio + (1-min_ratio) * np.random.rand(n)

    for i in range(n):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i,:h,:w] = misc.imresize(images[i], (h,w))
        images_new[i] = misc.imresize(images_new[i,:h,:w], (_h,_w))
        
    return images_new

def random_interpolate(images):
    _n, _h, _w = images.shape[:3]
    nd = images.ndim - 1
    assert _n % 2 == 0
    n = int(_n / 2)

    ratios = np.random.rand(n,*([1]*nd))
    images_left, images_right = (images[np.arange(n)*2], images[np.arange(n)*2+1])
    images_new = ratios * images_left + (1-ratios) * images_right
    images_new = images_new.astype(np.uint8)

    return images_new
    
def expand_flip(images):
    '''Flip each image in the array and insert it after the original image.'''
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, n=2*_n)
    images_new = np.stack([images, flip(images)], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new

def five_crop(images, size):
    _n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert h <= _h and w <= _w

    shape_new = get_new_shape(images, size, n=5*_n)
    images_new = []
    images_new.append(images[:,:h,:w])
    images_new.append(images[:,:h,-w:])
    images_new.append(images[:,-h:,:w])
    images_new.append(images[:,-h:,-w:])
    images_new.append(center_crop(images, size))
    images_new = np.stack(images_new, axis=1).reshape(shape_new)
    return images_new

def ten_crop(images, size):
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, size, n=10*_n)
    images_ = five_crop(images, size)
    images_flip_ = five_crop(flip(images), size)
    images_new = np.stack([images_, images_flip_], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new

def center_patch(images, landmarks, offset):
    _n, _h, _w = images.shape[:3]
    num_patches = 51
    # shape_new = get_new_shape(images, [offset, offset], n=num_patches*_n)
    images_new = np.ndarray((_n, num_patches, offset, offset, 3))
    # images_new = np.ndarray((_n, offset, offset, num_patches * 3))
    
    for i in range(_n):
        image = images[i].copy()
        cnt = 0
        for j in range(17*2, len(landmarks[i]), 2):
        # for j in range(72,94,2):    # eyes only
            if landmarks[i][j] < 0:
                landmarks[i][j] = 0
            if landmarks[i][j+1] < 0:
                landmarks[i][j+1] = 0
            if landmarks[i][j]-offset//2 >= 0:
                startx = landmarks[i][j]-offset//2
            else:
                startx = 0
            if landmarks[i][j+1]-offset//2 >= 0:
                starty = landmarks[i][j+1]-offset//2
            else:
                starty = 0
            if landmarks[i][j]+offset//2 <= image.shape[0]:
                endx = landmarks[i][j]+offset//2
            else:
                endx = image.shape[0]
                startx = image.shape[0]-offset
            if landmarks[i][j+1]+offset//2 <= image.shape[1]:
                endy = landmarks[i][j+1]+offset//2
            else:
                endy = image.shape[1]
                starty = image.shape[1]-offset

            _im = image[starty:endy,startx:endx]
            try:
                # images_new[i,:,:, cnt:cnt+3] = misc.imresize(_im, (offset, offset))
                images_new[i][cnt] = misc.imresize(_im, (offset,offset))
                # misc.imsave('orig.jpg', image)
                # misc.imsave('patch.jpg', _im)
            
            except ValueError as e:
                print(_im.shape, image.shape, startx, starty, endx, endy,
                 landmarks[i][j], landmarks[i][j+1], '\n')

                print(e)
                
                misc.imsave('orig.jpg', image)
                misc.imsave('patch.jpg', _im)
            
                import sys
                sys.exit(1)
            cnt += 1
    return images_new

# def eye_region(images, landmarks):
#     _n, _h, _w = images.shape[:3]
#     SIZE = [210, 210]
#     images_new = np.ndarray((_n, SIZE[0], SIZE[1], 3))
#     for i in range(_n):
#         image = images[i].copy()
#         ldmarks = [0 if ldmark < 0 else ldmark for ldmark in landmarks[i]]
#         leftx = ldmarks[34]
#         lefty = max(ldmarks[39], ldmarks[49])
#         # righty = max(ldmarks[81], ldmarks[93])
#         righty = ldmarks[115]
#         rightx = ldmarks[52]
#         images_new[i] = misc.imresize(image[lefty:righty,leftx:rightx],
#                         SIZE)
#     # if 0 in final.shape[:3]:
#     #     print(landmarks[34], landmarks[39], 
#     #           landmarks[49], landmarks[81],
#     #           landmarks[93], landmarks[52],
#     #          leftx, lefty, rightx, righty)
#     return images_new

def eye_region(images):
    _n, _h, _w = images.shape[:3]
    x1 = 20
    y1 = 64
    x2 = 256 - x1
    y2 = 150
    SIZE = [86, 216]
    images_new = np.ndarray((_n, SIZE[0], SIZE[1], 3))
    for i in range(_n):
        image = images[i].copy()
        images_new[i] = image[y1:y2,x1:x2]
    return images_new

def nose_region(images):
    _n, _h, _w = images.shape[:3]
    x1 = 128 - 30
    y1 = 80
    x2 = 128 + 30
    y2 = 200
    SIZE = [120, 60]
    images_new = np.ndarray((_n, SIZE[0], SIZE[1], 3))
    for i in range(_n):
        image = images[i].copy()
        images_new[i] = image[y1:y2,x1:x2]
    return images_new

def mouth_region(images):
    _n, _h, _w = images.shape[:3]
    x1 = 128 - 80
    y1 = 200
    x2 = 128 + 80
    y2 = 250
    SIZE = [50, 160]
    images_new = np.ndarray((_n, SIZE[0], SIZE[1], 3))
    for i in range(_n):
        image = images[i].copy()
        images_new[i] = image[y1:y2,x1:x2]
    return images_new

def eye_nose(images):
    eye = eye_region(images)
    nose = nose_region(images)
    return (eye,nose)

def global_eye_nose(images, is_train):
    eye = eye_region(images)
    nose = nose_region(images)
    global_images = resize(images, [128, 128])
    if is_train:
        global_images = random_crop(global_images, [112, 112])
    else:
        global_images = center_crop(global_images, [112, 112])
    return (global_images,eye,nose)

def random_distorted_crop(images, 
                    min_area,
                    max_aspect_ratio,
                    output_size):
    # 0. generate A (MIN_A 0.8), a within [0.8, 1.0]
    # 1. generate random square [h * a, w * a]
    # 2. generate random aspect ratio (MAX = 1.3)
    # 3. select location within image
    _n, _h, _w = images.shape[:3]
    h, w = tuple(output_size)
    shape_new = get_new_shape(images, output_size)

    alpha = np.random.rand(_n) * (1 - min_area) + min_area
    max_aspect = np.sqrt(max_aspect_ratio)
    min_aspect = 1./max_aspect
    aspect_ratio = np.random.rand(_n) * (max_aspect - min_aspect) + min_aspect
    aspect_ratio = np.minimum(np.maximum(aspect_ratio, alpha), 1./alpha)
   
    images_new = np.ndarray(shape_new)
    for i in range(_n):
        temp_h, temp_w = (_h * alpha[i], _w * alpha[i])
        temp_h_ar, temp_w_ar = (int(temp_h * aspect_ratio[i]), int(temp_w / aspect_ratio[i]))
        x = int(np.random.rand() * (_w - temp_w_ar))
        y = int(np.random.rand() * (_h - temp_h_ar))
        images_new[i] = misc.imresize(images[i, y:y+temp_h_ar, x:x+temp_w_ar]
, output_size)
    return images_new

register = {
    'resize': resize,
    'padding': padding,
    'random_crop': random_crop,
    'center_crop': center_crop,
    'random_flip': random_flip,
    'standardize': standardize_images,
    'random_shift': random_shift,
    'random_interpolate': random_interpolate,
    'random_rotate': random_rotate,
    'random_blur': random_blur,
    'random_noise': random_noise,
    'random_downsample': random_downsample,
    'expand_flip': expand_flip,
    'five_crop': five_crop,
    'ten_crop': ten_crop,
    'random_distorted_crop': random_distorted_crop,
    'patches': patches,
    'center_patch': center_patch,
    'eye_region': eye_region,
    'nose_region': nose_region,
    'mouth_region': mouth_region,
    'eye_nose': eye_nose,
    'global_eye_nose': global_eye_nose,
}

def resize_and_remove_bg(im, size, eyes_only=False):
    non_black_pixels = np.any(im > [10,10,10], axis=-1)
    min_x = np.min(np.where(non_black_pixels)[0])
    min_y = np.min(np.where(non_black_pixels)[1])
    max_x = np.max(np.where(non_black_pixels)[0])
    max_y = np.max(np.where(non_black_pixels)[1])
    cropped = im[min_x:max_x, min_y:max_y]
    cropped = misc.imresize(cropped, size=size)
    if eyes_only:
        x,y,_ = cropped.shape
        max_x = x
        max_y = int(np.round(y * 0.50))
        cropped = cropped[0:max_y, 0:max_x]
        return misc.imresize(cropped, size=size)
    else:
        return cropped

def load_files(images, config, is_training=False):
    image_list = []
    for image_path in images:
        image = misc.imread(image_path)
        image_list.append(image)
    return np.array(images_list)

def preprocess(images, config, is_training=False):
    # Load images first if they are file paths
    image_paths = images
    images = []
    # assert (config.channels==1 or config.channels==3)
    mode = 'RGB' if config.channels>=3 else 'I'
    for image_path in image_paths:
        image = np.array(Image.fromarray(imageio.imread(image_path)).resize((160,160)))

        images.append(image)

    images = np.stack(images, axis=0)
    # Process images
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        if 'center_patch' == proc_name:
             images = center_patch(images, ld, config.image_size[0])
        # elif 'eye_region' == proc_name:
        #     images = eye_region(images, landmarks)
        else:
            images = register[proc_name](images, *proc_args)
    #if len(images.shape) == 3:
    #    images = images[:,:,:,None]
    return images
        


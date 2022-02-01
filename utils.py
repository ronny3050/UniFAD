"""Utilities for training and testing
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
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
import numpy as np
from scipy import misc
import imp
import time
import math
import random
from datetime import datetime
import shutil
import facepy
from nntools.common.dataset_original import Dataset
from nntools.common.imageprocessing import *
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=1,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=None,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    return image, bbox_begin
    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox



class TFPreprocess:
    def __init__(self, size):
        self.graph = tf.Graph()
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        with self.graph.as_default():
           with self.sess.as_default():
                
                self.image = tf.placeholder(tf.float32, shape=[256,256,3], name='image')
                bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])

                self.distorted_image, self.distorted_bbox = distorted_bounding_box_crop(self.image, bbox)
                '''distorted_image.set_shape([None, None, 3])
                image_with_distorted_box = tf.image.draw_bounding_boxes(
                    tf.expand_dims(self.image, 0), distorted_bbox)
                num_resize_cases = 1
                distorted_image = apply_with_random_selector(
                    distorted_image,
                    lambda x, method: tf.image.resize_images(x, [size[0], size[1]], method), num_cases=num_resize_cases)
                distorted_image = tf.image.random_flip_left_right(distorted_image)
                distorted_image = tf.subtract(distorted_image, 0.5)
                self.distorted_image = tf.multiply(distorted_image, 2.0)
                '''
    def  __call__(self, images):
        
        results = []
        print(np.min(images), np.max(images))
        images = images/255.
        print(np.min(images), np.max(images))
        for i, image in enumerate(images):
           print('{}/{}'.format(i, len(images)))
           results.append(self.sess.run([self.distorted_image, self.distorted_bbox], feed_dict = {self.image: image}))
           print(results)
        return np.array(results)


def save_manifold(images, path, manifold_size=None, normalize=True):
    if normalize:
        images = (images+1.) / 2
    if manifold_size is None:
        manifold_size = image_manifold_size(images.shape[0])
    manifold_image = np.squeeze(merge(images, manifold_size))
    misc.imsave(path, manifold_image)
    return manifold_image

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

def merge(images, size):
    h, w, c = tuple(images.shape[1:4])
    manifold_image = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        manifold_image[j * h:j * h + h, i * w:i * w + w, :] = image

    if c == 1:
        return manifold_image
        # manifold_image = manifold_image[:,:,:,0]
    return manifold_image


def visualize_gradcam(figname, image, conv_output, conv_grad, gb_viz):
    output = conv_output           # [7,7,512]
    grads_val = conv_grad          # [7,7,512]
    # print("grads_val shape:", grads_val.shape)
    # gb_viz = np.repeat(gb_viz, repeats=3,  axis=-1)
    # print("gb_viz shape:", gb_viz.shape)
    # image = np.repeat(image, repeats=3,  axis=-1)
    # print("image shape:", image.shape)
    weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32) # [7,7]
    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = skimage.transform.resize(cam, (160,160), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    '''cam = np.float32(cam) + np.float32(img)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)'''
              
    fig = plt.figure()    
    ax = fig.add_subplot(141)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')
   
    ax = fig.add_subplot(142)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')    
    
    gb_viz = np.dstack((
            gb_viz[:, :, 0],
            gb_viz[:, :, 1],
            gb_viz[:, :, 2],
        ))       
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = fig.add_subplot(143)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')
    
    gd_gb = np.dstack((
            gb_viz[:, :, 0] * cam,
            gb_viz[:, :, 1] * cam,
            gb_viz[:, :, 2] * cam,
        ))            
    ax = fig.add_subplot(144)
    imgplot = plt.imshow(gd_gb)
    ax.set_title('guided Grad-CAM')

    # plt.show()
    plt.savefig(figname)
    plt.close()


def import_file(full_path_to_module, name='module.name'):
    
    module_obj = imp.load_source(name, full_path_to_module)
    
    return module_obj

def create_log_dir(config, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.log_base_dir), config.name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))

    return log_dir

def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    return learning_rate

def display_info(epoch, step, duration, watch_list):
    sys.stdout.write('[%d][%d] time: %2.2f' % (epoch+1, step+1, duration))
    for item in watch_list.items():
        if type(item[1]) in [float, np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [int, bool, np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')

def get_pairwise_score_label(score_mat, label):
    n = label.size
    assert score_mat.shape[0]==score_mat.shape[1]==n
    triu_indices = np.triu_indices(n, 1)
    if len(label.shape)==1:
        label = label[:, None]
    label_mat = label==label.T
    score_vec = score_mat[triu_indices]
    label_vec = label_mat[triu_indices]
    return score_vec, label_vec

def fuse_features(mu1, sigma_sq1, mu2, sigma_sq2):
    sigma_new = (sigma_sq1 * sigma_sq2) / (sigma_sq1 + sigma_sq2)
    mu_new = (sigma_sq2 * mu1 + sigma_sq1 * mu2) / (sigma_sq1 + sigma_sq2)
    return mu_new, sigma_new

def match_features(mu1, sigma_sq1, mu2, sigma_sq2):
    t1 = list(zip(mu1, sigma_sq1))
    t2 = list(zip(mu2, sigma_sq2))
    def metric(t1, t2):
        mu1, sigma_sq1 = tuple(t1)
        mu2, sigma_sq2 = tuple(t2)
        sigma_sq_sum = sigma_sq1 + sigma_sq2
        score = - np.sum(np.square(mu1 - mu2) / sigma_sq_sum) - np.sum(np.log(sigma_sq_sum))
        return score
    return facepy.protocol.compare_sets(t1, t2, metric)

"""Main training file for face recognition
"""
# MIT License
# 
# Copyright (c) 2019 Debayan Deb
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
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.special import expit

from nntools import tensorflow as tftools
from .. import utils as tfutils 
from .. import losses as tflosses
from .. import metric_loss_ops as mlosses
from .. import watcher as tfwatcher
from tensorflow.python.ops import math_ops

from tensorflow.contrib.tensorboard.plugins import projector

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ChimneyCNN:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes):
        '''
            Initialize the graph from scratch according config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                channels = config.channels
                h, w = config.image_size
                image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='image_batch')
                label_batch_placeholder = tf.placeholder(tf.int32, shape=[None], name='label_batch')
                learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
                keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                
                label_splits = tf.split(label_batch_placeholder, config.num_gpus)
                image_splits = tf.split(image_batch_placeholder, config.num_gpus)
                grads_splits = []
                split_dict = {}
                summaries = []
                average_dict = {}
                concat_dict = {}
                def insert_dict(_dict, k, v):
                    if k in _dict:
                        _dict[k].append(v)
                    else:
                        _dict[k] = [v]
                        
                for i in range(config.num_gpus):
                    scope_name = '' if i==0 else 'gpu_%d' % i
                    with tf.name_scope(scope_name):
                        with tf.variable_scope('', reuse=i>0):
                            with tf.device('/gpu:%d' % i):
                               
                                labels = tf.identity(label_splits[i], name='labels')
                                # Save the first channel for testing
                                images = tf.identity(image_splits[i], name='global_inputs')
                                self.inputs = images
                               
                                network = imp.load_source('network', config.network)
                                scores, final_feature = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
                                                                bottleneck_layer_size = config.embedding_size, 
                                                                weight_decay = config.weight_decay,
                                                                reuse=False,
                                                                model_version = config.model_version)  
                                self.embeddings = tf.nn.l2_normalize(final_feature, dim=1)                                                                                       
                                if i == 0:
                                    self.outputs = tf.identity(scores[-1], name='outputs')
                                self.total_loss = 0.0

                                # Early Sharing Loss
                                global_labels = tf.cast(tf.logical_not(tf.cast(tf.one_hot(labels, 1), tf.bool)), tf.float32)
                                early_loss = tflosses.sigmoid_cross_entropy_with_logits(scores[-1], global_labels)
                                insert_dict(average_dict, "early_loss", early_loss)

                                # Auxiliary Branch Losses
                                real_image_idx = tf.where(tf.equal(labels , 0))
                                adversarial_image_idx = tf.concat([real_image_idx, tf.where(tf.equal(labels , 1))], axis=0)
                                adversarial_labels = tf.concat([tf.zeros_like(real_image_idx), tf.ones_like(tf.where(tf.equal(labels , 1)))], axis=0)
                                digital_image_idx = tf.concat([real_image_idx, tf.where(tf.equal(labels , 2))], axis=0)
                                digital_labels = tf.concat([tf.zeros_like(real_image_idx), tf.ones_like(tf.where(tf.equal(labels , 2)))], axis=0)
                                physical_image_idx = tf.concat([real_image_idx, tf.where(tf.equal(labels , 3))], axis=0)
                                physical_labels = tf.concat([tf.zeros_like(real_image_idx), tf.ones_like(tf.where(tf.equal(labels , 3)))], axis=0)
                                adversarial_loss = tflosses.sigmoid_cross_entropy_with_logits(
                                    tf.squeeze(tf.gather(scores[0], adversarial_image_idx)), 
                                    tf.cast(tf.squeeze(adversarial_labels), tf.float32)
                                )
                                digital_loss = tflosses.sigmoid_cross_entropy_with_logits(
                                    tf.squeeze(tf.gather(scores[0], digital_image_idx)), 
                                    tf.cast(tf.squeeze(digital_labels), tf.float32)
                                )
                                physical_loss = tflosses.sigmoid_cross_entropy_with_logits(
                                    tf.squeeze(tf.gather(scores[0], physical_image_idx)), 
                                    tf.cast(tf.squeeze(physical_labels), tf.float32)
                                )

                                insert_dict(average_dict, "adv_loss", adversarial_loss)
                                insert_dict(average_dict, "dig_loss", digital_loss)
                                insert_dict(average_dict, "phy_loss", physical_loss)

                                G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNeXt")
                                Early_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNeXt/Common")
                                Adv_vars = Early_vars + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNeXt/AdversarialBranch")
                                Dig_vars = Early_vars + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNeXt/DigitalBranch")
                                Phy_vars = Early_vars + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNeXt/PhysicalBranch")
            
                                self.train_early_op = tf.train.AdamOptimizer(
                                    0.01, beta1=0.5, beta2=0.9
                                ).minimize(early_loss, var_list=G_vars)

                                self.train_adv_op = tf.train.AdamOptimizer(
                                    0.01, beta1=0.5, beta2=0.9
                                ).minimize(adversarial_loss, var_list=Adv_vars)

                                self.train_dig_op = tf.train.AdamOptimizer(
                                    0.01, beta1=0.5, beta2=0.9
                                ).minimize(digital_loss, var_list=Dig_vars)

                                self.train_phy_op = tf.train.AdamOptimizer(
                                    0.01, beta1=0.5, beta2=0.9
                                ).minimize(physical_loss, var_list=Phy_vars)

                                

                for k, v in average_dict.items():
                    v = tfutils.average_tensors(v)
                    average_dict[k] = v
                    tfwatcher.insert(k, v)
                    if "loss" in k:
                        summaries.append(tf.summary.scalar("losses/" + k, v))
                    elif "acc" in k:
                        summaries.append(tf.summary.scalar("acc/" + k, v))
                    else:
                        tf.summary(k, v)
                for k, v in concat_dict.items():
                    v = tf.concat(v, axis=0, name="merged_" + k)
                    concat_dict[k] = v
                    tfwatcher.insert(k, v)
                trainable_variables = [t for t in tf.trainable_variables()]


                self.update_global_step_op = tf.assign_add(global_step, 1)
                summaries.append(tf.summary.scalar("learning_rate", learning_rate_placeholder))
                self.summary_op = tf.summary.merge(summaries)

                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(trainable_variables, max_to_keep=None)


                # Keep useful tensors
                self.image_batch_placeholder = image_batch_placeholder
                self.label_batch_placeholder = label_batch_placeholder 
                self.learning_rate_placeholder = learning_rate_placeholder 
                self.keep_prob_placeholder = keep_prob_placeholder 
                self.phase_train_placeholder = phase_train_placeholder 
                self.global_step = global_step
                self.watch_list = tfwatcher.get_watchlist()
                self.config = config
                


    def train(self, image_batch, label_batch, learning_rate, keep_prob):
        feed_dict = {
                    self.image_batch_placeholder: image_batch,
                    self.label_batch_placeholder: label_batch,
                    self.learning_rate_placeholder: learning_rate,
                    self.keep_prob_placeholder: keep_prob,
                    self.phase_train_placeholder: True,
                    }
        _, _, _, _, wl, sm = self.sess.run([self.train_early_op, self.train_adv_op, self.train_dig_op, 
            self.train_phy_op, self.watch_list, self.summary_op], feed_dict = feed_dict)
        step = self.sess.run(self.global_step)

        return wl, sm, step
    
    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)
        

    def load_model(self, *args, **kwargs):
        tfutils.load_model(self.sess, *args, **kwargs)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
        self.inputs = self.graph.get_tensor_by_name('image_batch:0')
        self.outputs = tf.nn.sigmoid(self.graph.get_tensor_by_name('predicted:0'))
        # self.embeddings = self.graph.get_tensor_by_name('embeddings:0')


    def extract_feature(self, images, batch_size=512,
                        embeddings=False,
                        proc_func=None, 
                        verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.outputs.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        if embeddings:
            num_embedding = self.embeddings.shape[1]
            emb = np.ndarray((num_images, num_embedding), dtype=np.float32)
        start_time = time.time()
        times = []
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                times.append(time.time()-start_time)
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            inputs = proc_func(inputs) if proc_func else inputs
            feed_dict = {
                            self.inputs: inputs,
                            self.phase_train_placeholder: False,
                            self.keep_prob_placeholder: 1.0}
            if embeddings:
                result[start_idx:end_idx], emb[start_idx:end_idx] = self.sess.run([self.outputs, 
                    self.embeddings], feed_dict=feed_dict)
            else:
                result[start_idx:end_idx] = self.sess.run(self.outputs, feed_dict=feed_dict)
        if embeddings:
            return result, emb
        else:
            return result

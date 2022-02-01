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

class JointCNN:
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
                def insert_dict(k,v):
                    if k in split_dict: split_dict[k].append(v)
                    else: split_dict[k] = [v]
                        
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
                                prelogits, emb = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
                                                                bottleneck_layer_size = config.embedding_size, 
                                                                weight_decay = config.weight_decay,
                                                                reuse=False,
                                                                model_version = config.model_version)  
                                self.embeddings = tf.nn.l2_normalize(emb, dim=1)                                                                                       
                                if i == 0:
                                    self.outputs = tf.identity(prelogits, name='outputs')
                                self.total_loss = 0.0 
                                if 'sigmoid_cross_entropy' in config.losses.keys():
                                    self.final_labels = tf.cast(tf.logical_not(tf.cast(tf.one_hot(labels, 1), tf.bool)), tf.float32) 
                                    self.total_loss = tflosses.sigmoid_cross_entropy_with_logits(prelogits, self.final_labels)
                                    # self.emb_loss = tflosses.live_center_loss(self.embeddings, labels=labels)
                                    # self.emb_loss = tflosses.center_loss(self.embeddings, labels=labels, num_classes=2)
                                    # self.emb_loss = tflosses.deb_loss(self.embeddings, labels=labels)
                                    # self.total_loss = self.sigmoid_loss + 0.5 * self.emb_loss

                                if 'softmax_cross_entropy' in config.losses.keys():
                                    softmax_loss = tflosses.non_sparse_softmax_cross_entropy_with_logits(prelogits, labels, 2)
                                    tf.summary.scalar('softmax_loss', softmax_loss)
                                    self.total_loss = softmax_loss
                                if 'softmax_center' in config.losses.keys():
                                    softmax_loss = tflosses.softmax_cross_entropy_with_logits(prelogits, labels)
                                    center_loss = tflosses.center_loss(prelogits, labels, num_classes, coef=0.1)
                                    tf.summary.scalar('softmax_loss', softmax_loss)
                                    self.total_loss = softmax_loss + center_loss
                                if 'fixed_anchor' in config.losses.keys():
                                    print('FIXED ANCHOR LOSS')
                                    fixed_anchor_loss = tflosses.fixed_anchor(labels, self.embeddings)
                                    tf.summary.scalar('fixed_anchor_loss', fixed_anchor_loss)
                                    self.total_loss = fixed_anchor_loss
                                if 'constrastive_loss' in config.losses.keys():
                                    print('CONTRASTIVE LOSS')
                                    contrastive_loss = tflosses.contrastive_loss(labels, self.embeddings)
                                    tf.summary.scalar('contrastive_loss', contrastive_loss)
                                    self.total_loss = contrastive_loss
                                if 'triplet_loss' in config.losses.keys():
                                    print('TRIPLET LOSS')
                                    triplet_loss = tflosses.triplet_semihard_loss(labels, self.embeddings, **config.losses['triplet_loss'])
                                    #center_loss = tflosses.center_loss(self.embeddings, labels, 13, coef=0.2)
                                    #total_loss = triplet_loss
                                    tf.summary.scalar('triplet_loss',triplet_loss)
                                    #tf.summary.scalar('live_center_loss', center_loss)
                                    self.total_loss = triplet_loss

                                if 'deb_loss' in config.losses.keys():
                                    print('DEB LOSS')
                                    deb_loss = tflosses.deb_loss(labels, self.embeddings)
                                    tf.summary.scalar('deb_loss', deb_loss)
                                    self.total_loss = deb_loss
                                '''if 'aux_loss' in config.losses.keys():
                                    aux_loss = 0.4 * tflosses.sigmoid_cross_entropy_with_logits(aux_logits, tf.cast(labels[:,None], tf.float32))
                                    tf.summary.scalar('aux_loss', aux_loss)
                                    self.total_loss += aux_loss
                                '''
                                if 'center' in config.losses:
                                    print('CENTER LOSS')
                                    self.center_loss = tflosses.center_loss(prelogits, labels, 2)
                                    self.total_loss += self.center_loss

                                #self.total_loss = tflosses.sigmoid_cross_entropy_with_logits(prelogits, labels)     
                                #total_loss = tf.add_n(loss_list, name='total_loss')
                                grads_split = tf.gradients(self.total_loss, tf.trainable_variables())
                                grads_splits.append(grads_split)

                # Merge the splits
                grads = tfutils.average_grads(grads_splits)

                update_global_step_op = tf.assign_add(global_step, 1)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


               
                apply_gradient_op = tfutils.apply_gradient(tf.trainable_variables(), grads, config.optimizer,
                                        learning_rate_placeholder, config.learning_rate_multipliers)
                train_ops = [
                                apply_gradient_op, 
                                update_global_step_op
                            ] + update_ops


                
                tf.summary.scalar('total_loss', self.total_loss)
                # tfwatcher.insert('crs_loss', self.sigmoid_loss)
                # tfwatcher.insert('emb_loss', self.emb_loss)
                tfwatcher.insert('total_loss', self.total_loss)

                # var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
                # print('MODEL SIZE: ', sum(var_sizes) / (1024 ** 2), 'MB')
                
                train_op = tf.group(*train_ops)

                tf.summary.scalar('learning_rate', learning_rate_placeholder)
                summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)


                # Keep useful tensors
                self.image_batch_placeholder = image_batch_placeholder
                self.label_batch_placeholder = label_batch_placeholder 
                self.learning_rate_placeholder = learning_rate_placeholder 
                self.keep_prob_placeholder = keep_prob_placeholder 
                self.phase_train_placeholder = phase_train_placeholder 
                self.global_step = global_step
                self.train_op = train_op
                self.summary_op = summary_op
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
        _,  wl, sm = self.sess.run([self.train_op, self.watch_list, self.summary_op], feed_dict = feed_dict)
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
        self.embeddings = self.graph.get_tensor_by_name('embeddings:0')


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

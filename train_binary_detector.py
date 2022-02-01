"""Main training file for universal attack detection
"""
# MIT License
# 
# Copyright (c) 2022 Debayan Deb
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

import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from functools import partial
from scipy.special import expit
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
import utils
# import visualize
import facepy
from tensorflow.contrib.tensorboard.plugins import projector
from nntools.common.dataset import Dataset
from nntools.common.imageprocessing import preprocess, random_crop, patch_loc, random_flip
from nntools.tensorflow.networks import JointCNN
from facepy.metric import *
import evaluation
import seaborn as sns
import scipy.misc

from MulticoreTSNE import MulticoreTSNE as TSNE

def test(network, config, original_images, log_dir, step):
    output_dir = os.path.join(log_dir, 'samples')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    generated, mask = network.generate_images(original_images)
    utils.save_manifold(generated, os.path.join(output_dir, '{}_gen.jpg'.format(step)))
    utils.save_manifold(mask, os.path.join(output_dir, '{}_perturb.jpg'.format(step)))

def main(args):
    config_file = args.config_file
    # I/O
    config = utils.import_file(config_file, 'config')
    if args.name is not None:
        config.name = args.name

    loss_fn = list(config.losses.keys())[0]

    trainset = Dataset(config.train_dataset_path)
    testset = Dataset(config.test_dataset_path)
    
    network = JointCNN()
    network.initialize(config, trainset.num_classes)

    # Preprocessing functions

    # Initalization for running
    log_dir = utils.create_log_dir(config, config_file)
    summary_writer = tf.summary.FileWriter(log_dir, network.graph)
    if config.restore_model:
        network.restore_model(config.restore_model, config.restore_scopes)

    if not os.path.exists(os.path.join(log_dir, 'test')):
        os.makedirs(os.path.join(log_dir, 'test'))

    if not os.path.exists(os.path.join(log_dir, 'train')):
        os.makedirs(os.path.join(log_dir, 'train'))

    # Set up LFW test protocol and load images
    print('Loading images...')
    proc_func = lambda images: preprocess(images, config, True)
    trainset.start_batch_queue(config.batch_size, config.batch_format, proc_func=proc_func)
    
    best_tdr = 10000.0

    test_lives = np.where(testset.labels == 0)[0]
    idx = np.where(testset.labels == 0)[0]
    test_lives = idx[list(random.sample(range(0, len(idx)), 500))]
    test_spoofs = []
    for mat in range(1, len(np.unique(testset.labels))):
        idx = np.where(testset.labels == mat)[0]
        MIN_SPOOFS = min(500, len(idx))
        test_spoofs.extend(idx[list(random.sample(range(0, len(idx)), MIN_SPOOFS))])
    
    random_test_idx = test_lives.tolist() + test_spoofs

    #
    # Main Loop
    #
    print('\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n'\
        % (config.num_epochs, config.epoch_size, config.batch_size))
    global_step = 0
    start_time = time.time()
    tsne = TSNE(n_jobs=18, verbose=True)
    sns.set_context('paper')
    MAT_NAMES = ['real', 'AdvFaces', 'GFLM', 'DeepFool', 'Semantic']
    for epoch in range(config.num_epochs):

        if epoch == 0:
            testset_labels = testset.labels[random_test_idx]
            test_images = preprocess(testset.images[random_test_idx], config, False)
            test_gen_images = []
            for mat in range(0, 6):
                test_gen_images.extend(testset.images[testset.labels == mat][:7])
            test_gen_images = preprocess(test_gen_images, config, False)

        if global_step % config.evaluation_interval == 0 or global_step == 1:
                print('Testing on testing set')
                
                test_features, embeddings = network.extract_feature(test_images,
                    verbose=True, embeddings=True,
                    batch_size=64,)
                embeddings = tsne.fit_transform(embeddings)
                test_scores = expit(test_features)
                test_live_scores = test_scores[testset_labels == 0].tolist()
                test_spoof_scores = test_scores[testset_labels != 0].tolist()

                for mat in range(len(np.unique(testset.labels))):
                    idx = np.where(testset.labels[random_test_idx] == mat)[0]
                    sns.scatterplot(embeddings[idx, 0], embeddings[idx, 1], label=MAT_NAMES[mat], alpha=0.3)
                plt.savefig("{}/test/tsne_{}.png".format(log_dir, global_step))
                plt.clf()

                test_eer, test_eer_th, tdr = evaluation.eer(test_live_scores, test_spoof_scores)
                print('EER = {}%, TDR = {}%'.format(test_eer * 100, tdr * 100))
                
                eers = []
                for mat in range(1, len(np.unique(testset.labels))):
                    spoof = test_scores[testset_labels == mat].tolist()
                    eer, _, tdr_mat = evaluation.eer(test_live_scores, spoof)
                    print('{}: EER = {}%, TDR = {}%'.format(testset.images[testset.labels == mat][0].split('/')[-3], eer * 100, tdr_mat * 100))
                    eers.append(eer * 100)

                bins = np.linspace(0, 1, 100)
                plt.hist(np.array(test_spoof_scores), bins, alpha=.7, color='red', label='spoofs', density=True)
                plt.hist(np.array(test_live_scores), bins, alpha=.7, color='green', label='lives', density=True)
                plt.xlabel('Spoofness Score')
                plt.ylabel('Normalized Score Frequency')
                plt.legend(loc="upper right")
                plt.title('Spoof Detection Histogram')
                plt.savefig("{}/test/scores_{}.png".format(log_dir, global_step))
                plt.clf()

        # Training
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate = utils.get_updated_learning_rate(global_step, config)
            batch = trainset.pop_batch_queue()

            wl, sm, global_step = network.train(
                batch['images'],
                batch['labels'],
                learning_rate, config.keep_prob)

            wl['lr'] = learning_rate

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration, wl)
                summary_writer.add_summary(sm, global_step=global_step)

        network.save_model(log_dir, global_step)
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    parser.add_argument("--name", help="project_name",
                        type=str, default=None)
    args = parser.parse_args()
    main(args)
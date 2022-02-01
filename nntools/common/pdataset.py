"""Data fetching with pandas
"""
# MIT License
# 
# Copyright (c) 2018 Yichun Shi
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
import time
import math
import random
import shutil
from multiprocessing import Process, Queue

import h5py
import numpy as np
import pandas as pd

# TODO: Everything

class DataClass(object):
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.indices = np.array(indices)
        self.label = label
        return

    def random_pair(self):
        return np.random.permutation(self.indices)[:2]

    def random_samples(self, num_samples_per_class, exception=None):
        indices_temp = list(self.indices[:])
        if exception is not None:
            indices_temp.remove(exception)
            assert len(indices_temp) > 0
        # Sample indices multiple times when more samples are required than present.
        indices = []
        iterations = int(np.ceil(1.0*num_samples_per_class / len(indices_temp)))
        for i in range(iterations):
            sample_indices = np.random.permutation(indices_temp)
            indices.append(sample_indices)
        indices = np.concatenate(indices, axis=0)[:num_samples_per_class]
        return indices

    def build_clusters(self, cluster_size):
        permut_indices = np.random.permutation(self.indices)
        cutoff = (permut_indices.size // cluster_size) * cluster_size
        clusters = np.reshape(permut_indices[:cutoff], [-1, cluster_size])
        clusters = list(clusters)
        if permut_indices.size > cutoff:
            last_cluster = permut_indices[cutoff:]
            clusters.append(last_cluster)
        return clusters




class Dataset(object):

    def __init__(self, path=None):

        if path is not None:
            self.init_from_path(path)
        else:
            self.data = pd.DataFrame([], columns=['paths', 'abspaths', 'labels', 'names'])
            self.prefix = ''


        # self.DataClass = DataClass
        self.index_queue = None
        self.index_worker = None
        self.batch_queue = None
        self.batch_workers = None
       

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        return self.data[key]

    def _delitem(self, key):
        self.data.__delitem__(key)

    def init_from_path(self, path):
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        elif ext == '.txt':
            self.init_from_list(path)
        elif ext == '.hdf5':
            self.init_from_hdf5(path)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder, .txt or .hdf5 file' % path)
        # print('%d images of %d classes loaded' % (len(self.images), self.num_classes))

    def init_from_folder(self, folder):
        folder = os.path.abspath(os.path.expanduser(folder))
        class_names = os.listdir(folder)
        class_names.sort()
        paths = []
        labels = []
        names = []
        for label, class_name in enumerate(class_names):
            classdir = os.path.join(folder, class_name)
            if os.path.isdir(classdir):
                images_class = os.listdir(classdir)
                images_class.sort()
                images_class = [os.path.join(class_name,img) for img in images_class]
                paths.extend(images_class)
                labels.extend(len(images_class) * [label])
                names.extend(len(images_class) * [class_name])
        abspaths = [os.path.join(folder,p) for p in paths]
        self.data = pd.DataFrame({'paths': paths, 'abspaths': abspaths, 
                                            'labels': labels, 'names': names})
        self.prefix = folder

    

    def init_from_list(self, filename, folder_depth=2):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        abspaths = [os.path.abspath(line[0]) for line in lines]
        paths = ['/'.join(p.split('/')[-folder_depth:]) for p in abspaths]
        if len(lines[0]) == 2:
            labels = [int(line[1]) for line in lines]
            names = [str(lb) for lb in labels]
        elif len(lines[0]) == 1:
            names = [p.split('/')[-folder_depth] for p in abspaths]
            _, labels = np.unique(names, return_inverse=True)
        else:
            raise ValueError('List file must be in format: "fullpath(str) \
                                        label(int)" or just "fullpath(str)"')

        self.data = pd.DataFrame({'paths': paths, 'abspaths': abspaths, 
                                            'labels': labels, 'names': names})
        self.prefix = abspaths[0].split('/')[:-folder_depth]

       
    def init_crossval_folder(self, folder):
        folder = os.path.expanduser(folder)
        paths = []
        labels = []
        fold = []
        splits = os.listdir(folder)
        splits.sort()
        for fd, splitdir in enumerate(splits):
            splitdir = os.path.join(folder, splitdir)
            class_names = os.listdir(splitdir)
            class_names.sort()
            for label, class_name in enumerate(class_names):
                classdir = os.path.join(splitdir, class_name)
                if os.path.isdir(classdir):
                    images_class = os.listdir(classdir)
                    images_class.sort()
                    images_class = [os.path.join(classdir,img) for img in images_class]
                    paths.extend(images_class)
                    labels.extend(len(images_class) * [label])
                    fold.extend(len(images_class) * [fd])

        self.data = pd.DataFrame({'paths': paths, 'abspaths': paths, 
                            'labels': labels, 'names': names, 'fold': fold})
        


    @property
    def num_classes(self):
        return len(self.data['labels'].unique())

    @property
    def classes(self):
        return self.data['labels'].unique()

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def loc(self):
        return self.data.loc       

    @property
    def iloc(self):
        return self.data.iloc


    def import_features(self, listfile, features):
        assert self.images.shape[0] == features.shape[0]
        with open(listfile, 'r') as f:
            images = f.readlines()
        img2idx = {}
        for i, image in enumerate(images):
            img2idx[os.path.abspath(image.strip())] = i
        self.features = np.ndarray((features.shape[0], features.shape[1]), dtype=np.float)
        for i in range(self.images.shape[0]):
            self.features[i] = features[img2idx[os.path.abspath(self.images[i])]]
        return self.features
        

    def import_column(self, column, listfile, data, allow_nan=False):
        with open(listfile, 'r') as f:
            paths = f.readlines()
            paths = ['/'.join(os.path.abspath(p.strip()).split('/')[-2:]) for p in paths]
        
        img2idx = {}
        for i, path in enumerate(self.data['paths']):
            img2idx[path] = i
        
        column_data = [None] * len(data)
        for i, value in enumerate(data):
            path = paths[i]
            column_data[img2idx[path]] = value
        
        self.data[column] = column_data
            
        if not allow_nan:
            assert not self.data[column].isnull().any(), \
                'ImportColumn: {} instances not found'.format(self.data[column].isnull().sum())

    #
    # Data Loading
    #

    def random_samples_from_class(self, label, num_samples, exception=None):
        # indices_temp = self.class_indices[label]
        indices_temp = list(np.where(self.data['labels'].values == label)[0])
        
        if exception is not None:
            indices_temp.remove(exception)
            assert len(indices_temp) > 0
        # Sample indices multiple times when more samples are required than present.
        indices = []
        iterations = int(np.ceil(1.0*num_samples / len(indices_temp)))
        for i in range(iterations):
            sample_indices = np.random.permutation(indices_temp)
            indices.append(sample_indices)
        indices = list(np.concatenate(indices, axis=0)[:num_samples])
        return indices

    def init_index_queue(self, batch_format):
        if self.index_queue is None:
            self.index_queue = Queue()
        
        if batch_format['sampling'] in ['random_samples', 'random_samples_with_mates']:
            size = self.data.shape[0]
            index_queue = np.random.permutation(size)[:,None]
        else:
            raise ValueError('IndexQueue: Unknown batch_format: {}!'.format(batch_format))
        for idx in list(index_queue):
            self.index_queue.put(idx)


    def get_batch_indices(self, batch_format):
        ''' Get the indices from index queue and fetch the data with indices.'''
        indices_batch = []
        sampling = batch_format['sampling']
        batch_size = batch_format['size']

        if sampling =='random_samples':
            while len(indices_batch) < batch_size:
                indices_batch.extend(self.index_queue.get(block=True, timeout=30)) 
            assert len(indices_batch) == batch_size

        elif sampling == 'random_classes':
            num_classes = batch_format['num_classes']
            assert batch_size % num_classes == 0
            num_samples_per_class = batch_size // num_classes
            idx_classes = np.random.permutation(self.classes)[:num_classes]
            indices_batch = []
            for c in idx_classes:
                indices_batch.extend(self.random_samples_from_class(c, num_samples_per_class))

        elif sampling == 'random_samples_with_mates':
            num_seeds = batch_format['num_seeds']
            num_samples_per_class = batch_size // num_seeds
            assert batch_size % num_seeds == 0
            while len(indices_batch) < batch_size:
                seed_idx = self.index_queue.get(block=True, timeout=30)
                assert len(seed_idx) == 1
                seed_idx = seed_idx[0]
                c= self.data['labels'][seed_idx]
                indices_batch.extend([seed_idx] + \
                    list(self.random_samples_from_class(c, num_samples_per_class-1, exception=seed_idx)))
            assert len(indices_batch) == batch_size
            
        else:
            raise ValueError('get_batch: Unknown batch_format: {}!'.format(batch_format))

        return indices_batch

    def get_batch(self, batch_format):

        indices = self.get_batch_indices(batch_format)
        batch = {}
        for column in self.data.columns:
            batch[column] = self.data[column].values[indices]

        return batch

    # Multithreading preprocessing images
    def start_index_queue(self, batch_format):

        if not batch_format['sampling'] in ['random_samples', 'random_samples_with_mates']:
           return

        self.index_queue = Queue()
        def index_queue_worker():
            while True:
                if self.index_queue.empty():
                    self.init_index_queue(batch_format)
                time.sleep(0.01)
        self.index_worker = Process(target=index_queue_worker)
        self.index_worker.daemon = True
        self.index_worker.start()

    def start_batch_queue(self, batch_format, proc_func=None, maxsize=1, num_threads=3):
        if self.index_queue is None:
            self.start_index_queue(batch_format)

        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker(seed):
            np.random.seed(seed)
            while True:
                batch = self.get_batch(batch_format)
                if proc_func is not None:
                    batch['images'] = proc_func(batch['abspaths'])
                self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)
    
    def pop_batch_queue(self, timeout=60):
        return self.batch_queue.get(block=True, timeout=timeout)
      
    def release_queue(self):
        if self.index_queue is not None:
            self.index_queue.close()
        if self.batch_queue is not None:
            self.batch_queue.close()
        if self.index_worker is not None:
            self.index_worker.terminate()   
            del self.index_worker
            self.index_worker = None
        if self.batch_workers is not None:
            for w in self.batch_workers:
                w.terminate()
                del w
            self.batch_workers = None


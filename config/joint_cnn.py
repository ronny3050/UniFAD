''' Config Proto '''

import sys
import os

####### INPUT OUTPUT #######

# The interval between writing summary
summary_interval = 100

train_dataset_path = 'config/train_joint.txt'
test_dataset_path = 'config/train_chimney.txt' # for visualizing JointCNN perf. on each attack type

# The folder to save log and model
log_base_dir = 'log'
name = 'jointCNN'

# Target image size for the input of network
image_size = [160, 160]
image_mode = 'rgb'

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    # ['resize', (48,56)],
    # ['resize', image_size],
    ['random_flip'],
    # ['center_crop', (112, 112)],
    # ['standardize', 'mean_scale'],
    # ['random_crop', (28,28)],
    # ['random_downsample', 0.5],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    # ['resize', image_size],
    # ['center_crop', (112, 112)],
    # ['patches', (28,28)],
    # ['resize', (64, 64)],
    ['standardize', 'mean_scale'],
    
]

# Number of GPUs
num_gpus = 1

####### NETWORK #######

# Auto alignment network
localization_net = None

# The network architecture
network = "nets/joint_cnn.py"

# Model version, only for some networks
model_version = "test"

# Number of dimensions in the embedding space
embedding_size = 1


####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("MOM", {'momentum': 0.9})

# Number of samples per batch
batch_size = 180

# The structure of the batch
batch_format = 'random_even_classes:4'


# Number of batches per epoch
epoch_size = 500

# Number of epochs
num_epochs = 200

evaluation_interval = epoch_size

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.01
learning_rate_schedule = {
    0:      1 * lr,
}

# Multiply the learning rate for variables that contain certain keywords
learning_rate_multipliers = {
    # 'ConditionalLoss/weights': ('MOM', 100.0)
    # 'SplitSoftmax/threshold_': 1.0,
    # 'BinaryLoss/weights': 100.,
    # 'LocalizationNet/': 1e-3,
}

# Restore model
restore_model = None
# restore_model = 'log/binary_cnn/occl/20210131-205614'

# Keywords to filter restore variables, set None for all
restore_scopes = None

# Weight decay for model variables
weight_decay = 5e-4

# Keep probability for dropouts
keep_prob = 1.0


####### LOSS FUNCTION #######

# Scale for the logits
losses = {
    # 'softmax': {},
    # 'cosine': {'gamma': 'auto'},
    # 'angular': {'m': 4, 'lamb_min':5.0, 'lamb_max':1500.0},
    # 'split': {'gamma': 'auto', 'm': 0.7, "weight_decay": 5e-4},
    # 'am_softmax': {'scale': 'auto', 'm': 5.0, 'alpha': 'auto'},
    # 'stochastic': {'coef_kl_loss': 1e-3},
    # 'norm': {'alpha': 1e-2},
    # 'triplet': {'margin': 1.0},
    'sigmoid_cross_entropy': {},
    # 'occl': {},
    # 'one_class_contrastive': {},
    #'softmax_cross_entropy': {}
    # 'deb_loss': {},
}


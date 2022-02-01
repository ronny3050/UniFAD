from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

model_params = {
    'basic': ([0, 0, 0, 0], [16, 32, 64, 128]),
    'test': ([0, 1, 2, 3, 2], [64, [64,128], [128,256], [256,512], [256,512]], 3, 16),
    #'test': ([0, 2, 3, 4, 3], [64, [64,128], [128,256], [256,512], [256,512]], 3, 16),
    #'test': ([0, 3, 4, 6, 3], [64, [64,128], [128,256], [256,512], [256,512]], 3, 16),
    #'test': ([0, 0, 0, 0, 0], [64, [64,128], [128,256], [256,512], [256,512]], 3, 16),
    '50':   ([0, 3, 4, 6, 3], [64, [128,256], [256,512], [512,1024], [1024, 2048]], 7, 32),
    '101':  ([0, 3, 4, 23, 3], [64, [128,256], [256,512], [512,1024], [1024, 2048]], 7, 32),
    '152':  ([0, 3, 8, 36, 3], [64, [128,256], [256,512], [512,1024], [1024, 2048]], 7, 32),
}

batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}   

batch_norm_params_last = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 10e-8,
    # force in-place updates of mean and variance estimates
    'center': False,
    # not use beta
    'scale': False,
    # not use gamma
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}

activation = tf.nn.relu

# Convolution with special initialization
def convolution(net, num_kernels, kernel_size, groups=1, stride=1, padding='SAME'):
    assert num_kernels % groups == 0, '%d %d' % (kernel_size, groups)
    stddev = math.sqrt(2/(kernel_size*kernel_size*num_kernels/groups)) 
    if groups==1:
        return slim.conv2d(net, num_kernels, kernel_size=kernel_size, stride=stride, padding=padding,
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                biases_initializer=None)
    else:
        num_kernels_split = int(num_kernels / groups)
        input_splits = tf.split(net, groups, axis=3)
        output_splits = [slim.conv2d(input_split, num_kernels_split, 
                kernel_size=kernel_size, stride=stride, padding=padding,
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                biases_initializer=None) for input_split in input_splits]
        return tf.concat(output_splits, axis=3)

def residual_block(net, num_kernels, cardinality, stride=1, reuse=None, scope=None):
    with tf.variable_scope(scope, 'block', [net], reuse=reuse):
        net = convolution(net, num_kernels[0], kernel_size=1, groups=1, stride=1, padding='SAME')
        net = convolution(net, num_kernels[0], kernel_size=3, groups=cardinality, stride=stride, padding='SAME')
        print(net.shape)
        with slim.arg_scope([slim.conv2d], activation_fn=None):
            net = convolution(net, num_kernels[1], kernel_size=1, groups=1, stride=1, padding='SAME')
    return net
             
def conv_module(net, num_res_layers, num_kernels, trans_kernel_size=3, trans_stride=2,
                     use_se=False, reuse=None, scope=None):
    with tf.variable_scope(scope, 'conv', [net], reuse=reuse):
        net = slim.conv2d(net, num_kernels, 
                kernel_size=trans_kernel_size, stride=trans_stride, padding='SAME',
                weights_initializer=slim.xavier_initializer())
        shortcut = net
        for i in range(num_res_layers):
            # num_kernels_sm = int(num_kernels / 2)
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            # net = slim.conv2d(net, num_kernels, kernel_size=1, stride=1, padding='SAME',
            #     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            #     biases_initializer=None)
            print('| ---- block_%d' % i)
            if use_se:
                net = se_module(net)
            net = net + shortcut
            shortcut = net
    return net           

# def conv_module(net, num_res_layers, num_kernels, cardinality, stride, reuse = None, scope = None):
#     with tf.variable_scope(scope, 'conv', [net], reuse=reuse):
#         # Use convolution for the first shortcut
#         shortcut = convolution(net, num_kernels[1], kernel_size=1, stride=stride, padding='SAME')
#         for i in range(num_res_layers):
#             stride = stride if i==0 else 1
#             net = residual_block(net, num_kernels, cardinality, stride, 
#                             reuse=reuse, scope='block_%d' % i)
#             print('| ---- block_%d' % i)
#             net = activation(net + shortcut)
#             shortcut = net

#     return shortcut

# def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=512, 
#             weight_decay=1e-4, reuse=None, model_version=None):
#     with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected],
#                         activation_fn=activation,
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params=batch_norm_params):
#         with tf.variable_scope('ResNeXt', [images], reuse=reuse):
#             with slim.arg_scope([slim.batch_norm, slim.dropout],
#                                 is_training=phase_train):
#                 print('input shape:', [dim.value for dim in images.shape])
                
#                 model_version = 'test' if model_version ==None else model_version
#                 num_layers, num_kernels, kernel_size, cardinality = model_params[model_version]
#                 net = convolution(images, num_kernels[0], kernel_size=kernel_size, groups=1, stride=2, padding='SAME')
#                 print('module_1 shape:', [dim.value for dim in net.shape])

#                 net = conv_module(net, num_layers[1], num_kernels[1], cardinality, stride=1, scope='conv2')
#                 print('module_2 shape:', [dim.value for dim in net.shape])
                
#                 net = conv_module(net, num_layers[2], num_kernels[2], cardinality, stride=2, scope='conv3')
#                 print('module_3 shape:', [dim.value for dim in net.shape])

#                 net = conv_module(net, num_layers[3], num_kernels[3], cardinality, stride=2, scope='conv4')
#                 print('module_4 shape:', [dim.value for dim in net.shape])

#                 net = conv_module(net, num_layers[4], num_kernels[4], cardinality, stride=2, scope='conv5')
#                 print('module_5 shape:', [dim.value for dim in net.shape])
                
#                 net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool5')
#                 net = slim.flatten(net)
#                 net = slim.fully_connected(net, 256, scope='PreBottleneck',
#                                         # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
#                                         # weights_initializer=tf.constant_initializer(0.),
#                                         weights_initializer=slim.xavier_initializer())
#                 net = slim.fully_connected(net, 1, scope='Bottleneck',
#                                         # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
#                                         # weights_initializer=tf.constant_initializer(0.),
#                                         weights_initializer=slim.xavier_initializer(), 
#                                         activation_fn=None)

#     return net


def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=512, 
            weight_decay=1e-4, reuse=None, model_version=None):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('ResNeXt', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('input shape:', [dim.value for dim in images.shape])
                net = conv_module(images, 0, 16, scope='global_conv1')
                print('module_1 she:', [dim.value for dim in net.shape])
                net = conv_module(net, 0, 32, scope='global_conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])
                net = conv_module(net, 0, 64, scope='global_conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])
                net = conv_module(net, 0, 128, scope='global_conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])
                
                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool5')
                feat = slim.flatten(net)
                '''feat = slim.fully_connected(net, 64, scope='PreBottleneck',
                #                         # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                #                         # weights_initializer=tf.constant_initializer(0.),
                                        weights_initializer=slim.xavier_initializer(),
                                        activation_fn=None, normalizer_fn=None)'''
                net = slim.fully_connected(feat, 1, scope='Bottleneck',
                                        # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        # weights_initializer=tf.constant_initializer(0.),
                                        weights_initializer=slim.xavier_initializer(), 
                                        activation_fn=None, normalizer_fn=None)

    return net, feat

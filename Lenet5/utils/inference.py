# coding: utf-8

import tensorflow as tf

IMAGE_SIZE = 28  # 输入维度大小
NUM_CHANNELS = 1  # 通道数
CONV1_SIZE = 5  # 卷积核维度
CONV1_KERNEL_NUM = 32  # 卷积核个数
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512  # 全连接层节点
OUTPUT_SIZE = 10  # 输出节点（分类数目）

def create_placeholder():
    with tf.name_scope('input_data'):
        X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name = 'X')
        Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name = 'Y')
#         image_shaped_input = tf.reshape(X, [-1, 28, 28, 1])
        tf.summary.image('X_images', X, 10) # 记录输入图像
    return X, Y

def get_weight(shape, regularizer):  # 权重初始化  # tf.get_variable('weight', shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name = 'weight')
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(W))
    tf.summary.histogram('weight', W) # 记录权重
    return W

def get_biasis(shape):  # 偏置初始化
    b = tf.Variable(tf.zeros(shape), name = 'biasis')
    tf.summary.histogram('biasis', b)
    return b
    
def conv2d(X, W):  # 卷积操作，X: [N, H, W, C]
    return tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool(X):  # 最大池化，2x2，步长2
    return tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def reshaped(pool):  # 全连接层维度拉直
    pool_shape = pool.get_shape().as_list()
#     print(pool_shape)
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  # reshaped = tf.contrib.layers.flatten(pool2)
    reshaped = tf.contrib.layers.flatten(pool)
#     reshaped = tf.reshape(pool, [pool_shape[0], nodes])
    return reshaped, nodes

def forward(X, train = True, regularizer = 0.0001):  # 前向传播，train决定训练时使用dropout    
    with tf.variable_scope('conv32_5x5'):  # 第一层
        conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM], regularizer)
        conv1_b = get_biasis([CONV1_KERNEL_NUM])
        conv1 = conv2d(X, conv1_w)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
        pool1 = max_pool(relu1)
        
    with tf.variable_scope('conv64_5x5'):  # 第二层
        conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM], regularizer)
        conv2_b = get_biasis([CONV2_KERNEL_NUM])
        conv2 = conv2d(pool1, conv2_w)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
        pool2 = max_pool(relu2)
        
    with tf.variable_scope('fc_512'):  # 全连接层
        reshaped1, nodes = reshaped(pool2)
        fc1_w = get_weight([nodes, FC_SIZE], regularizer)
        fc1_b = get_biasis([FC_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped1, fc1_w) + fc1_b)
        if train == True: fc1 = tf.nn.dropout(fc1, 0.5)  # 训练时dropout
        
    with tf.variable_scope('output_10'):  # 输出全连接层
        o_w = get_weight([FC_SIZE, OUTPUT_SIZE], regularizer)
        o_b = get_biasis([OUTPUT_SIZE])
        y = tf.matmul(fc1, o_w) + o_b
    
    return y
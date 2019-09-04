
# coding: utf-8

# ##### Lenet-5:
# <img src="Lenet5.png">
# 

# In[2]:


# inference
import tensorflow as tf
import numpy as np
import os

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
        tf.summary.image('X_images', X, 10)
    return X, Y

def get_weight(shape, regularizer):  # 权重初始化  # tf.get_variable('weight', shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name = 'weight')
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(W))
    tf.summary.histogram('weight', W)
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


# In[5]:


from tensorflow.examples.tutorials.mnist import input_data

STEPS = 100  # 根据所有样本的代数停止
INTERATIONS = 10000  # 根据迭代次数停止训练
BATCH_SIZE = 256  # 批量大小
LEARNING_RATE_BASE = 0.005  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减系数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均系数
MODEL_SAVE_PATH = 'G:/deeplearning/Lenet5/model/'
MODEL_NAME = 'Lenet-5_model'

if not os.path.exists(MODEL_SAVE_PATH): os.mkdir(MODEL_SAVE_PATH)

mnist = input_data.read_data_sets("G:/MNIST_data/", one_hot = True)
def backward(mnist):
    tf.reset_default_graph()
    X, Y = create_placeholder()
    y = forward(X, True)
    
    global_step = tf.Variable(0, trainable = False, name = 'train_step')
    learning_rate = tf.train.exponential_decay(  # 学习率指数衰减，
        LEARNING_RATE_BASE, 
        global_step, 
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY, 
        staircase = True)
    tf.summary.scalar('learning_rate', learning_rate)
    
    with tf.name_scope('optimizer'):
        with tf.name_scope('loss'):
            mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = Y))  
            # tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(Y, 1))
            tf.add_to_collection('losses', mean_loss)  # loss = mean_loss + tf.add_n(tf.get_collection('losses'))
            loss = tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('mean_loss', mean_loss)
            tf.summary.scalar('loss_l2', loss)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  # 滑动平均
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, ema_op]):  # 训练时同步更新滑动平均值
            train_op = tf.no_op(name = 'train')
        
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
        tf.summary.scalar('train', accuracy)
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 允许显存增长，（显存不足）
    saver = tf.train.Saver()
    with tf.Session(config = config) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
#         step = sess.run(global_step)
        n_batch = int(mnist.train.num_examples / BATCH_SIZE)
        for epoch in range(STEPS): # range(step, STEPS)
            epoch_loss = 0
            epoch_acc = 0
            for i in range(n_batch):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                reshaped_xs = np.reshape(xs, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
                _, loss_value, acc, step, summary = sess.run([train_op, loss, accuracy, global_step, merged], feed_dict = {X:reshaped_xs, Y:ys})
#                 print(step)
                epoch_loss += loss_value/n_batch
                epoch_acc += acc/n_batch
                writer.add_summary(summary, step)
                print('INFO: After %d training iteration(s), loss is %.5f, batch_train accuracy is %.4f' % (step, loss_value, acc))
                if (i+1) % 100 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
                    print('%s-%d.ckpt saved!' % (MODEL_NAME, step))
                    
                if step >= INTERATIONS: return
                
#                     print('INFO: After %d training iteration(s), loss is %.5f, batch_train accuracy is %.4f' % (step, loss_value, acc))
#             if epoch % 10 == 0:
            print('After %d training epoch(s)(X%d), loss is %.5f, train accuracy is %.4f' % (epoch+1, n_batch, epoch_loss, epoch_acc))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME+'.ckpt-{:d}x{:d}').format(epoch+1, n_batch))
            print('%s-%dx%d saved!' % (MODEL_NAME, epoch+1, n_batch))
#             sess.run(global_step.assign(epoch))
#             print('%s-%d saved!' % (MODEL_NAME, epoch))
#             if step >= INTERATIONS: break
    
    
backward(mnist)


# In[10]:


# 评估准确率
import tensorflow as tf
import time
import numpy as np

TEST_INTERVAL_SECS = 5  # 间隔

def evaluation(mnist_image, mnist_label, num_examples):  # mnist.test.images, minst.test.num_examples
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        Y = tf.placeholder(tf.float32, [None, 10])
        y = forward(X, False, None)
        
        em = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)  # 滑动平均
        em_restore = em.variables_to_restore()
        saver = tf.train.Saver(em_restore)
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    steps = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshape_x = np.reshape(mnist_image, [num_examples, 28, 28, 1])
                    acc_ = sess.run(accuracy, feed_dict = {X:reshape_x, Y:mnist_label})
                    print('After %s training step(s), evaluation accuracy is %.4f' % (steps, acc_))
                else:
                    print('No checkpoint file found!')
                    return
            time.sleep(TEST_INTERVAL_SECS)

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("G:/MNIST_data/", one_hot = True)
evaluation(mnist.test.images, mnist.test.labels, mnist.test.num_examples)


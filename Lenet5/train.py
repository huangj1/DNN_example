"""
Example usage:
    ./train.py --model_dir='./path/' \
        --num_train_steps=numbers \
"""

import tensorflow as tf
import numpy as np
import os
from utils import inference

STEPS = 100  # 根据所有样本的代数停止
#INTERATIONS = 10000  # 根据迭代次数停止训练
BATCH_SIZE = 256  # 批量大小
LEARNING_RATE_BASE = 0.005  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减系数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减系数
#MODEL_SAVE_PATH = 'G:/deeplearning/Lenet5/model/'  # 模型保存路径
MODEL_NAME = 'Lenet-5_model'  # 模型名称

flags = tf.app.flags
flags.DEFINE_string('model_dir', 'G:/deeplearning/Lenet5/model/', 'Path to output model directory where event and checkpoint files will be written.')
flags.DEFINE_integer('num_train_steps', 10000, 'Number of train steps.')

FLAGS = flags.FLAGS

MODEL_SAVE_PATH = FLAGS.model_dir
INTERATIONS = FLAGS.num_train_steps
if not os.path.exists(MODEL_SAVE_PATH): os.mkdir(MODEL_SAVE_PATH)

def backward(mnist):
    tf.reset_default_graph()
    X, Y = inference.create_placeholder()
    y = inference.forward(X, True)
    
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
    config.gpu_options.allow_growth = True  # 允许显存增长，（显存不足时）
    saver = tf.train.Saver()
    with tf.Session(config = config) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        n_batch = int(mnist.train.num_examples / BATCH_SIZE)
        for epoch in range(STEPS): # range(step, STEPS)
            epoch_loss = 0
            epoch_acc = 0
            for i in range(n_batch):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                reshaped_xs = np.reshape(xs, [BATCH_SIZE, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS])
                _, loss_value, acc, step, summary = sess.run([train_op, loss, accuracy, global_step, merged], feed_dict = {X:reshaped_xs, Y:ys})
#                 print(step)
                epoch_loss += loss_value/n_batch
                epoch_acc += acc/n_batch
                writer.add_summary(summary, step)
                print('INFO: After %d training iteration(s), loss is %.5f, batch_train accuracy is %.4f' % (step, loss_value, acc))
                if step % 100 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME+ '.ckpt'), global_step = global_step)
                    print('%s-%d.ckpt saved!' % (MODEL_NAME, step))
                    
                if step >= INTERATIONS:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME+ '.ckpt'), global_step = global_step)
                    return
            print('After %d training epoch(s)(X%d), loss is %.5f, train accuracy is %.4f' % (epoch+1, n_batch, epoch_loss, epoch_acc))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME+'.ckpt-{:d}x{:d}').format(epoch+1, n_batch))
            print('%s.ckpt-%dx%d saved!' % (MODEL_NAME, epoch+1, n_batch))
			
def main(self):
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("G:/MNIST_data/", one_hot = True)
	backward(mnist)
	
if __name__ == '__main__':
	tf.app.run()
'''
python valuation.py
'''

# 评估准确率
import tensorflow as tf
import time
import numpy as np
from utils import inference

TEST_INTERVAL_SECS = 5  # 间隔

def evaluation(mnist_image, mnist_label, num_examples):  # mnist.test.images, minst.test.num_examples
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        Y = tf.placeholder(tf.float32, [None, 10])
        y = inference.forward(X, False, None)
        
        em = tf.train.ExponentialMovingAverage(0.99)  # 滑动平均
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
			
def main(): 
	#MOVING_AVERAGE_DECAY = 0.99
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("G:/MNIST_data/", one_hot = True)
	evaluation(mnist.test.images, mnist.test.labels, mnist.test.num_examples)
	
if __name__ == '__main__':
	MODEL_SAVE_PATH = 'G:/deeplearning/Lenet5/model/'
	main()
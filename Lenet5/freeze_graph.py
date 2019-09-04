import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(input_path, out_graph): # 模型固化，ckpt->pb
	output_node_names = 'accuracy/accuracy'
	ckpt = tf.train.get_checkpoint_state(input_path)
	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', clear_devices=True)
		
	with tf.Session() as sess:
		saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复图并得到数据
		output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
		sess = sess,
		input_graph_def = sess.graph_def,  # tf.get_default_graph().as_graph_def()
		output_node_names = output_node_names.split(',')  # 如果有多个输出节点，以逗号隔开
		)
		
		with tf.gfile.GFile(out_graph, 'wb') as f:  # 保存模型
			f.write(output_graph_def.SerializeToString())  # 序列化输出
		print('%d ops in the final graph.' % len(output_graph_def.node))  # 得到当前图有几个操作节点
		
def freeze_graph_test(pb_path): # 读取pb
	with tf.Graph().as_default():
		output_graph_def = tf.GraphDef()
		with open(pb_path, 'rb') as f:
			output_graph_def.ParseFromString(f.read())
			tf.import_graph_def(output_graph_def, name='')
		
		#with tf.Session() as sess:
        #    sess.run(tf.global_variables_initializer())
 
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入
        #    input_image_tensor = sess.graph.get_tensor_by_name("input:0")
		
flags = tf.app.flags

flags.DEFINE_string('input_path', 'G:/deeplearning/Lenet5/model/', 'Path to trained checkpoint')
flags.DEFINE_string('out_graph', 'G:/deeplearning/Lenet5/model/frozen_model.pb', 'Path to write outputs')

FLAGS = flags.FLAGS

def main(self):
	input_path = FLAGS.input_path
	out_graph = FLAGS.out_graph
	freeze_graph(input_path, out_graph)
	
if __name__ == '__main__':
	tf.app.run()
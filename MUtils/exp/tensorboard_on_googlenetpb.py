import tensorflow as tf

graph_def = tf.GraphDef()
with open('/mnt/d/Data/testgen/googlenet/googlenet.pb', 'rb') as f:
	graph_def.ParseFromString(f.read())

LOGDIR='/home/marcelo/tensorflow/Scripts/MUtils/exp'
graph = tf.import_graph_def(graph_def, return_elements=['fc_biaser'])

with tf.Session() as sess:
	train_writer=tf.summary.FileWriter(LOGDIR)
	train_writer.add_graph(sess.graph)
	tf.global_variables_initializer().run()

import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import InferenceEngine
import tensorflow as tf

inf = InferenceEngine.InferenceEngine(arch="HW")
g = tf.get_default_graph()
print(g)
gdef = g.as_graph_def()
for node in gdef.node:
	print(node.name)
LOGDIR='/home/marcelo/'
with tf.Session() as sess:
	train_writer=tf.summary.FileWriter(LOGDIR)
	train_writer.add_graph(sess.graph)
	tf.global_variables_initializer().run()

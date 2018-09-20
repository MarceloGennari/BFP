# This just shows how to get a graph object from a graphdef

import tensorflow as tf

graph_def = tf.GraphDef()
with open('/mnt/d/Data/OmnitekInceptionV1MangledWeights.pb', 'rb') as f:
	graph_def.ParseFromString(f.read())

LOGDIR='/mnt/d/Data/PBFiles/'

tf.train.write_graph(graph_def, LOGDIR, "OmnitekInceptionV1MangledWeights.pbtext", as_text=True)
#with tf.Session() as sess:
#	train_writer = tf.summary.FileWriter(LOGDIR)
#	train_writer.add_graph(graph_def)

#for node in graph_def.node:
#	print(node.name)

graph = tf.import_graph_def(graph_def)
print(graph)

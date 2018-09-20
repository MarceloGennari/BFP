
import tensorflow as tf

graph_def = tf.GraphDef()
with open('/mnt/d/Data/testgen/googlenet/googlenet.pb', 'rb') as f:
	graph_def.ParseFromString(f.read())

graph = tf.import_graph_def(graph_def, return_elements=[n.name for n in graph_def.node])

node = graph_def.node[0]
print(node.attr['value'].tensor)
#print(graph)
#for node in graph_def.node:
#	if node.op=="Conv2D":
#		print(node.attr["shift_values"])

with tf.Session() as sess:
	print(graph[0].tensor_content.eval())

import google.protobuf.text_format as text_format
import tensorflow as tf

graph_def = tf.GraphDef()
with open('/mnt/d/Data/PBFiles/OmnIncV1Weights_to_float.pbtext', "rb") as f:
	text_format.Merge(f.read(), graph_def)


graph = tf.import_graph_def(graph_def)

import yaml
import numpy as np
import tensorflow as tf

bfp_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/bfp_out.so')
slim = tf.contrib.slim

class QuantEngine:

	def __init__(self, path_to_yaml):
				

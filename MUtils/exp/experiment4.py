import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import InferenceEngine
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

int8_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/int8_out.so')
qnt = int8_out_module.int8_out

inf = InferenceEngine.InferenceEngine()
weights = inf.get_weights()
init_assign_fn = slim.assign_from_checkpoint_fn('/mnt/d/Data/Inception/inception_v1_noBatch.ckpt', weights)
with tf.Session() as sess:
	init_assign_fn(sess)
	w, sc = qnt(weights[20])
	we = w.eval()
	sca = sc.eval()
scaling = np.power(2,sca)
print(we.shape)
print(we)
print(sca)
print(np.array(np.multiply(scaling,we)))

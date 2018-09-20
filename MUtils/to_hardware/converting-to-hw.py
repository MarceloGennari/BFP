import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import os
import InferenceEngine
import tensorflow as tf
import numpy as np
import math

slim = tf.contrib.slim

pertensor_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/pertensor_out.so')
qnt = pertensor_out_module.pertensor_out

map_to_hw = [1,3,0,4,2,5]
names = ['conv1', 'conv2r', 'conv2', 'i3a', 'i3b', 'i4a', 'i4b', 'i4c', 'i4d', 'i4e', 'i5a', 'i5b','fc']
order = ['3r', '5r', '1', '5', '3', 'pp']
inception_blocks = ['i3a', 'i3b', 'i4a', 'i4b','i4c','i4d', 'i4e', 'i5a', 'i5b']
non_inception_blocks = ['conv1', 'conv2r', 'conv2', 'fc']

inf = InferenceEngine.InferenceEngine()
weights = inf.get_weights()
init_assign_fn = slim.assign_from_checkpoint_fn('/mnt/d/Data/Inception/inception_v1_noBatch.ckpt',weights)

def quantize_to_hardware(weights, last_downstream_scale, index, k, index_w):
	w, sc = qnt(weights[4*k])
	we = w.eval()
	sca = sc.eval()
	bias = weights[4*k+1].eval()

	scale = np.amax(sca)
	closest_to_base2 = int(round(math.log2(scale)))
	shift_value = math.pow(2, closest_to_base2)
	downstream_scale = last_downstream_scale*scale/shift_value
	bias_scaled = bias * downstream_scale
	# To adjust for the 3x3 and 5x5 convolution confusion, create a 5x5 one and assign
	if(index%6==0 and index!=0):
		newW = np.zeros((5,5,we.shape[2], we.shape[3]))
		newW[1:4, 1:4, :, : ] = we
		Weights = newW

	print("Maximum of weight: " + str(np.amax(np.abs(we))))
	print("Scaling Factor: "+ str(scale))
	print("Scaling Factor Shift: " + str(shift_value))
	print("Downstream Scale: " + str(downstream_scale))
	print("Maximum of Bias: " + str(np.amax(np.abs(bias_scaled))))
	last_downstream_scale = downstream_scale
	return last_downstream_scale	

def get_index_from_pb(k):
	which_inception = None	
	index_w = None
	if k<3:
		dirname = names[k]
	if k >=3 and k!= 57:
		which_inception = (k-3)%6
		index_w = math.floor((k-3)/6)
		dirname = names[3+index_w]
		k = map_to_hw[which_inception] + 3 + index_w*6
	if k == 57:
		dirname = names[len(names)-1]
	return dirname, k, index_w, which_inception



with tf.Session() as sess:
  init_assign_fn(sess)
  last_downstream_scale = 1
  for k in range(58):
    index = k
    dirname, k, index_w, which_inception = get_index_from_pb(k)
	
    # This tells which block the weights are part of
    if dirname in inception_blocks:
      if which_inception==0:
        print("Inception block " + str(index_w) + ": " + dirname)
      dirname, k, index_w, which_inception = get_index_from_pb(k)
      print("	" + order[which_inception])
      last_downstream_scale = quantize_to_hardware(weights, last_downstream_scale, index, k, index_w)
      print("")
    if dirname not in inception_blocks:
      print("Not in Inception Block " + dirname)
      last_downstream_scale = quantize_to_hardware(weights, last_downstream_scale, index, k, index_w)

    







































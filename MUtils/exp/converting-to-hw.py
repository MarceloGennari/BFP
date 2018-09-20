import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import os
import InferenceEngine
import tensorflow as tf
import numpy as np
import math

slim = tf.contrib.slim

int8_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/int8_out.so')
qnt = int8_out_module.int8_out

map_to_hw = [1,3,0,4,2,5]
names = ['conv1', 'conv2r', 'conv2', 'i3a', 'i3b', 'i4a', 'i4b', 'i4c', 'i4d', 'i4e', 'i5a', 'i5b','fc']
order = ['3r', '5r', '1', '5', '3', 'pp']
inf = InferenceEngine.InferenceEngine()
weights = inf.get_weights()
init_assign_fn = slim.assign_from_checkpoint_fn('/mnt/d/Data/Inception/inception_v1_noBatch_biasScaled.ckpt',weights)
with tf.Session() as sess:
	init_assign_fn(sess)
	for k in range(58):
		index = k
		if k<3:
			dirname = names[k]
		if k >=3 and k!= 57:
			which_inception = (k-3)%6
			index_w = math.floor((k-3)/6)
			dirname = names[3+index_w]
			k = map_to_hw[which_inception] + 3 + index_w*6
		if k == 57:
			dirname = names[len(names)-1]
		print(dirname)
		w, sc = qnt(weights[4*k])
		we = w.eval()
		sca = sc.eval()
		scaling = np.power(2, sca)
		we = np.array(np.multiply(scaling, we))
		
		bias, s = qnt(weights[4*k+1])
		b = bias.eval()
		s2 = s.eval()
		s2 = np.array(s2, dtype=float)
		scaling = np.power(2,s2)
		be = np.array(np.multiply(scaling, b))
		# Putting everything to int
		Weights = np.array(we, dtype=int)
		if(k==57):
			Weights = np.array(we[0,0, :, 0:1000], dtype = int)
		ScalingWeights = np.array(sca[0,0,0,:], dtype=int)
		# To adjust for the 3x3 and 5x5 convolution confusion, create a 5x5 one and assign
		if(index%6==0 and index!=0):
			newW = np.zeros((5,5,Weights.shape[2], Weights.shape[3]))
			newW[1:4, 1:4, :, : ] = Weights
			Weights = newW
		if(k==57):
			ScalingWeights = np.array(sca[0,0,0,0:1000], dtype = int)
		Bias = np.array(be, dtype=int)
		if(k==57):
			Bias = np.array(be[0:1000], dtype=int)
		ScalingBias = np.array(s2, dtype=int)
		if(k==57):
			ScalingBias = np.array(s2[0:1000], dtype=int)
		# Here the scale of the bias is incorporated in the scale of the product of the input and the weight
		Scaling = ScalingWeights-ScalingBias
		print(Weights.shape)
		dirPath = '/mnt/d/Data/WeightsHW/'
		if k>=3 and k!=57:
			fileName = order[which_inception]
		else:
			fileName = ''
		np.save(dirPath+'w_'+dirname+fileName, Weights)
		np.save(dirPath+'b_'+dirname+fileName, Bias)
		np.save(dirPath+'sb_'+dirname+fileName, ScalingBias)
		np.save(dirPath+'sw_'+dirname+fileName, ScalingWeights)
		np.save(dirPath+'s_'+dirname+fileName, Scaling)
		if k>=3 and k!=57:
			k = index

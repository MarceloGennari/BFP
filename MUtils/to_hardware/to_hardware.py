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

def scale_to_127(weight, bias):
	w, sc = qnt(weight)
	w_scaled = w.eval()
	scale = np.amax(sc.eval())
	b = bias.eval()
	closest_to_base2 = int(round(math.log2(scale)))
	shift_real_value = math.pow(2, closest_to_base2)
	return w_scaled, b, scale, shift_real_value

#####
# Names of the layers and convolutions
#####
names = ['conv1', 'conv2r', 'conv2', 'i3a', 'i3b', 'i4a', 'i4b', 'i4c', 'i4d', 'i4e', 'i5a', 'i5b','fc']
order = ['3r', '5r', '1', '5', '3', 'pp'] # This is the branch orders in each inception module

map_to_pb = [1, 3, 0, 4, 2, 5]
inception_blocks = ['i3a', 'i3b', 'i4a', 'i4b','i4c','i4d', 'i4e', 'i5a', 'i5b']
test_blocks = inception_blocks #['i3a', 'i3b', 'i4a']
non_inception_blocks = ['conv1', 'conv2r', 'conv2', 'fc']

inf = InferenceEngine.InferenceEngine()
variables = inf.get_weights()
init_assign_fn = slim.assign_from_checkpoint_fn('/mnt/d/Data/Inception/inception_v1_noBatch.ckpt',variables)

#####
# This function is going to return the index in the checkpoint that corresponds to the pb file
#####
def get_index(k):
	if k <3 or k==57:
		weight_index = 4*k
		bias_index = 4*k+1
	if k>=3 and k!=57:
		which_inception = math.floor((k-3)/6) # This tells which of the inception blocks it is
		which_operation = (k-3)%6 	      # This tells which operation in the block is being computed
		k = map_to_pb[which_operation]+3+which_inception*6
		weight_index = 4*k
		bias_index = 4*k+1
	return weight_index, bias_index

def get_weights_bias_scale(counter):
	weight_index, bias_index = get_index(counter)
	weightCkpt = variables[weight_index]
	biasCkpt = variables[bias_index]
	weight, bias, scale, shift_real_value = scale_to_127(weightCkpt, biasCkpt)
	return weight, bias, scale, shift_real_value	

def scale_bias_to_1(bias_scaled, shift_real_value):
	if bias_scaled <=1:
		while bias_scaled < 0.5:
			bias_scaled*=2
			shift_real_value/=2
		return shift_real_value
	if bias_scaled >1:
		while bias_scaled >1:
			bias_scaled/=2
			shift_real_value *=2
		return shift_real_value	

def save_to_disk(weight, bias, shift, fileName):
	dirPath = '/mnt/d/Data/WeightsHW_2/'
	np.save(dirPath + 'w_' + fileName, np.array(weight, dtype=int))
	np.save(dirPath + 'b_' + fileName, np.array(bias*127, dtype=int))
	np.save(dirPath + 's_' + fileName, np.array(np.ones(weight.shape[weight.ndim-1])*int(math.log2(shift)), dtype=int))

def assign_to_checkpoint(weight, bias, shift_real_value, counter):
	weight_index, bias_index = get_index(counter)
	variables[weight_index].assign(weight/shift_real_value).eval()
	variables[bias_index].assign(bias).eval()	

def save_to_checkpoint():
	saver = tf.train.Saver()
	save_path = saver.save(sess, "./hardware_variables.ckpt")
	print("Model saved in path: %s" % save_path)

with tf.Session() as sess:
	init_assign_fn(sess)
	last_downstream_scale = 1
	counter = 0
	for k in range(len(names)): # This is going to run to each "block" at a time
		if names[k] in inception_blocks and names[k] in test_blocks:
			print("This the inception block: " + names[k])
			per_op = {}
			scales_of_block = np.array([])
			down_scales_of_block = np.array([])
			weights_of_block =[]
			for inblock_index in range(6):
				weight, bias, scale, shift_real_value = get_weights_bias_scale(counter)
				weight = weight*shift_real_value/scale
				
				while np.amax(np.abs(weight))>127:
					shift_real_value/=2
					weight/=2
							
				bias_scale = shift_real_value/shift_real_value
				downstream_scale = last_downstream_scale * bias_scale
				bias_scaled = bias*downstream_scale
				scales_of_block = np.append(scales_of_block, scale)
				down_scales_of_block = np.append(down_scales_of_block, bias_scale)
				weights_of_block.append(weight)

			#	weight_index, bias_index = get_index(counter)
			#	variables[bias_index].assign(bias_scaled).eval()
			#	counter+=1
			#	continue
				# Adjusting the bias
#				shift_real_value = scale_bias_to_1(np.amax(np.abs(bias_scaled)), shift_real_value)
#				bias_scale = scale/shift_real_value
#				downstream_scale = last_downstream_scale*bias_scale
#				bias_scaled = bias*downstream_scale

				assign_to_checkpoint(weight, bias_scaled, shift_real_value, counter)
				
				per_op.update({order[inblock_index]: {'scale': scale} })
				per_op[order[inblock_index]]['shift'] = shift_real_value
				per_op[order[inblock_index]]['bias_scale'] = downstream_scale
				last_downstream_scale = downstream_scale
				
				# Adjusting for the fact that this 3x3 kernel should be a 5x5
				if order[inblock_index] == '5':
					tmp = np.zeros((5, 5, weight.shape[2], weight.shape[3]))
					tmp[1:4, 1:4, :, :] = weight
					weight = tmp
				
				print(order[inblock_index]+ ": ")
				print("Maximum of weight: " + str(np.amax(np.abs(weight))))
				print("Scaling Factor: " + str(scale))
				print("Shift Real Value: " + str(shift_real_value))
				print("Downstream Scaling: " + str(downstream_scale))
				print("Maximum of Bias Scaled: " + str(np.amax(np.abs(bias_scaled))))
				counter+=1
				fileName = names[k] + order[inblock_index]
				save_to_disk(weight, bias_scaled, shift_real_value, fileName)
			print(scales_of_block)	
			print(down_scales_of_block)

			print("")

		if names[k] in non_inception_blocks:
			print("This is the non inception block: " + names[k])
			weight, bias, scale, shift_real_value = get_weights_bias_scale(counter)
			bias_scale = scale/shift_real_value
			downstream_scale = last_downstream_scale*bias_scale
			bias_scaled = bias*downstream_scale
			
			# Adjusting the bias
#			shift_real_value = scale_bias_to_1(np.amax(np.abs(bias_scaled)), shift_real_value)
#			bias_scale = scale/shift_real_value
#			downstream_scale = last_downstream_scale*bias_scale
#			bias_scaled = bias*downstream_scale

			assign_to_checkpoint(weight, bias_scaled, shift_real_value, counter)

			# Adjusting for the fact that the last FC should be 1000 instead of 1001 classes
			if names[k] == 'fc':
				weight = weight[0, 0, :, 0:1000]
				bias_scaled = bias_scaled[0:1000]	
			counter+=1

			last_downstream_scale = downstream_scale
			print("Maximum of weight: " + str(np.amax(np.abs(weight))))
			print("Scaling Factor: " + str(scale))
			print("Shift Real Value: " + str(shift_real_value))
			print("Downstream Scaling: " + str(downstream_scale))
			print("Maximum of Bias Scaled: " + str(np.amax(np.abs(bias_scaled))))
			print("")
			fileName = names[k]
			save_to_disk(weight, bias_scaled, shift_real_value, fileName)

	save_to_checkpoint()

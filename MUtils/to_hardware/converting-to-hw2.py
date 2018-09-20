import numpy as np

names = ['conv1', 'conv2r', 'conv2', 'i3a', 'i3b', 'i4a', 'i4b', 'i4c', 'i4d', 'i4e', 'i5a', 'i5b', 'fc']
order = ['3r', '5r', '1', '5', '3', 'pp']
branch0 = ['1']
branch1 = ['5r', '5']
branch2 = ['3r', '3']
branch3 = ['pp']
inception_modules = ['i3a','i3b', 'i4a', 'i4b', 'i4c', 'i4d', 'i4e', 'i5a', 'i5b']
non_inception_modules = ['conv1', 'conv2r', 'conv2', 'fc']
# NOTICE THAT IN EACH INCEPTION MODULE THERE ARE TWO BRANCHES THAT HAVE ONLY ONE CONVOLUTION
# BRANCH 0: 1
# BRANCH 1: 5R + 5
# BRANCH 2: 3R + 3
# BRANCH 3: PP
# SO BOTH IN PP AND IN 1 THERE IS ONLY ONE CONVOLUTION LAYER

# The idea of this script is to make sure all of the inception modules and all of the weights and biases
# have consistent scaling.

def get_weights(module, inc_block=''):
	dirname = '/mnt/d/Data/WeightsHW/'
	weights       = np.load(dirname + 'w_' + module + inc_block + '.npy')
	bias          = np.load(dirname + 'b_' + module + inc_block + '.npy')
	scale_bias    = np.load(dirname + 'sb_'+ module + inc_block + '.npy')
	scale_weights =	np.load(dirname + 'sw_'+ module + inc_block + '.npy')
	print('Recover weights correctly')
	return weights, bias, scale_weights, scale_bias

for module in names:
	# if module is not equal to conv1, conv2r, conv2 or fc
	if module in inception_modules:
		print("Inception Module: " + module)
		for inc_block in order:
			if inc_block in branch0:
				print("	Inception branch 0: " + inc_block)
			if inc_block in branch1:
				print("	Inception branch 1: " + inc_block)
			if inc_block in branch2:
				print("	Inception branch 2: " + inc_block)
			if inc_block in branch3:
				print("	Inception branch 3: " + inc_block)
	elif module not in inception_modules:
		print("Non inception module: " + module)
		w, b, sw, sb = get_weights(module)
		print(sb)



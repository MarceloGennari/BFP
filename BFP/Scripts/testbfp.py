import tensorflow as tf
import numpy as np
from ../MUtils import ImgProc

bfp_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/bfp_out.so')

oneDim = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]		# [Dimensions(12)]
twoDim = [[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]]		# [Dimensions(2), Dimensions(6)]
threeDim = [[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]],[[17,18,19,20],[21,22,23,24]]]	# [Dimensions(2), Dimensions(2), Dimensions(3)]

threeDim = np.array(threeDim)
shp = threeDim.shape
print("Dim in Python " +  str(shp))
shp = list(shp)
shp.insert(0, 1)
shp = tuple(shp)
print("New Dim in Python " + str(shp))
threeDim = np.reshape(threeDim, shp)
print("Dim in Python " + str(threeDim.shape))

with tf.Session(''):
	#print(bfp_out_module.bfp_out(oneDim).eval())
	#print(bfp_out_module.bfp_out(twoDim).eval())
	print(bfp_out_module.bfp_out(threeDim).eval())

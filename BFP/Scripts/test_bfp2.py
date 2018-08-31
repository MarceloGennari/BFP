import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/')
sys.path.insert(0, '/home/marcelo/git/models/research/slim/preprocessing/')

import preprocessing_factory as pp
from MUtils.img_proc import ImgProc

bfp_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/bfp_out.so')

proc = pp.get_preprocessing('inception')

######
# Loading Image to test
######
data_dir = "/mnt/d/Data/ILSVRC2012/ILSVRC2012_img_val/"
img_pth = data_dir+"ILSVRC2012_val_00000028.jpeg"
img = ImgProc(img_pth, 1).preprocess()
imgplot = plt.imshow(img[0])
plt.show()

with tf.Session(''):
	print(img)
	res1 = bfp_out_module.bfp_out(img, ShDepth=3, MWidth=2, EWidth=2).eval()
	imgplot = plt.imshow(res1[0])
	plt.show()
	print(res1[0])

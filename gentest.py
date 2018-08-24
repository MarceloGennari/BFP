from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from PIL import Image
from nets import inception
from MUtils.sysnet_labels import Label
from MUtils.img_proc import ImgProc
slim = tf.contrib.slim

##############
# Creating Parser
##############
parser = argparse.ArgumentParser(description='Setting variables')
parser.add_argument("-d", "--debug", help="Prints information to help debugging", action="store_true")
parser.add_argument("-m", "--mantissa", help="Specifies the width of the mantissa", type=int, default=23)
parser.add_argument("-x", "--exponent", help="Specifies the width of the exponent", type=int, default=8)
parser.add_argument("-s", "--sharedexp", help="Specifies the depth of shared exponent", type=int, default=16)
parser.add_argument("-b", "--batch", help="Specifies the value of the batch" , type=int, default=1)
parser.add_argument("-e", "--epoch", help="Specifies the number of epochs", type=int, default=1)
parser.add_argument("-o", "--original", help="Specifies whether to use original or altered network", action="store_true")
args = parser.parse_args()

debug = args.debug
alter = not args.original
batch_size = args.batch
epochs = args.epoch
m_w = args.mantissa
e_w = args.exponent
s_w = args.sharedexp

##############
# Defining the Model
##############
height, width = 224, 224
num_channels = 3
path_to_ckpt = "/mnt/d/Data/"
model = 'InceptionV1'
num_clss = 0
if model == 'InceptionV1':
	num_clss = 1001
	path_to_ckpt = path_to_ckpt + "Inception/inception_v1.ckpt"
elif model == 'VGG16':
	num_clss = 1000
	path_to_ckpt = path_to_ckpt + "VGG/vgg_16.ckpt"
elif model == 'VGG19':
	num_clss = 1000
	path_to_ckpt = path_to_ckpt + "VGG/vgg_19.ckpt"
elif model == 'ResNetV150':
	num_clss = 1000
	path_to_ckpt = path_to_ckpt + "ResNet/resnet_v1_50.ckpt"

X = tf.placeholder(tf.float32, shape=[None, height, width, num_channels])
with slim.arg_scope(inception.inception_v1_arg_scope()):
	logits, end_points = inception.inception_v1(X, num_classes=num_clss, is_training=False, m_w=m_w, e_w=e_w, s_w=s_w, alter=alter, debug=debug)

predictions = end_points["Predictions"]	# The end model is stored here
variables_to_restore = slim.get_variables_to_restore()
init_assign_fn = slim.assign_from_checkpoint_fn(path_to_ckpt, variables_to_restore)

##############
# Loading all images
##############
data_dir = "/mnt/d/Data/ILSVRC2012/ILSVRC2012_img_val/"
batch = np.empty([batch_size, height, width, num_channels])

for i in range(batch_size):
	path = data_dir + "ILSVRC2012_val_" + str(i+1).zfill(8) + ".jpeg"
	batch[i] = ImgProc(path, i+1).preprocess()

##############
# Loading validation ground truth
##############
gr_truth_dir = "/mnt/d/Data/ILSVRC2012/"
fil = open(gr_truth_dir + "ILSVRC2012_validation_ground_truth.txt", "r")
ground_truth = np.array(fil.read().splitlines(), int)

##############
# Loading Mapping to sysnet
##############
lab_map = "/mnt/d/Data/"
mTru = Label(lab_map+"ILSVRCMap.txt")
MPred = Label(lab_map+"Inception/MappingSysnet.txt")
if model == 'InceptionV1':
	mPred = Label(lab_map + "Inception/MappingSysnet.txt")

##############
# Performing Inference
##############
batch_s = int(batch_size/epochs)
predicted_classes = np.empty([batch_size, num_clss], int)
for i in range(epochs):
	it = int(i*batch_s)
	with tf.Session() as sess:
		init_assign_fn(sess)
		predictions_val = predictions.eval(feed_dict={X: batch[it:it+batch_s]})
		predicted_classes[it:it+batch_s] = np.argsort(predictions_val, axis=1)

##############
# Printing results
##############
right1 = 0
false1 = 0
right5 = 0
false5 = 0
pr = debug
for i in range(predicted_classes.shape[0]):
	if pr:
		print("For image " + str(i+1) + " the top five are: ")
		for j in range(5):
			print("	" + str(j+1) + ": " + mPred.getHumanLabel(predicted_classes[i][num_clss-1-j]-1))
	if mTru.isTop1(ground_truth[i]-1, mPred, predicted_classes[i]):
		right1+=1
	else:
		false1+=1
	if mTru.isTop5(ground_truth[i]-1, mPred, predicted_classes[i]):
		right5+=1
	else:
		false5+=1	
print("Mantissa Width: " + str(args.mantissa))		 
print("Exponent Width: " + str(args.exponent))
print("Top 1: "+ str(right1) + " correct and " + str(false1) + " false")
print("Top 5: "+ str(right5) + " correct and " + str(false5) + " false")

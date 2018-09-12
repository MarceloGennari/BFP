from PIL import Image
import numpy as np
import tensorflow as tf
import preprocessing_factory as pp

class ImgProc:
	def __init__(self, filePath, index=-1, pr=False):
		self.image = Image.open(filePath)
		self.index = index
		self.pr = pr
		# Deals with images that are in the CMYK format
		if self.image.mode == 'CMYK':
			print("Converting to RGB")
			self.image = self.image.convert('RGB')
		self.image = np.asarray(self.image)

	def check_colour_channel(self, pr = False):
		# Dealing with imges that do not have colour channel
		if(len(self.image.shape)!=3):
			if(self.pr):
				print("Image number " + str(self.index) + " doesn't have a colour channel")
			self.image = np.array(self.image).transpose()
			self.image = np.array([self.image]*3).transpose()
		
		assert self.image.shape[2] == 3, "not 3 channels for image " + str(self.index)
	
	def preprocess(self, arch='inception', pr = False, add_batch = True):
		self.check_colour_channel()	
		proc = pp.get_preprocessing(arch)
		self.image = tf.convert_to_tensor(self.image)
		self.image = proc(self.image, 224, 224)
		with tf.Session('') as sess:
			self.image = sess.run(self.image)
		ready_img = self.image
		if add_batch:
			ready_img = np.reshape(self.image, [1, 224, 224, 3])
		return ready_img	

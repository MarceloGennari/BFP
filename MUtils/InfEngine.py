import yaml
import numpy as np
import tensorflow as tf
from nets import inception
from nets import vgg
from img_proc import ImgProc
from sysnet_labels import Label

bfp_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/bfp_out.so')
slim = tf.contrib.slim

class InfEngine:

	def __init__(self, path_to_yaml):
		self.path_to_yaml = path_to_yaml
		self.__read_yaml()
		self.__set_up()
		self.__get_sysnet()

	def inference(self):
		batch_s = int(self.config["batch_size"]/self.config["epochs"])
		self.predicted_classes = np.empty([self.config["batch_size"],self.num_clss], int)
		for i in range(self.config["epochs"]):
			it = int(i*batch_s)
			with tf.Session() as sess:
				self.init_assign_fn(sess)
				predictions_val =self.predictions.eval(feed_dict={self.X: self.batch[it:it+batch_s]})
				self.predicted_classes[it:it+batch_s] = np.argsort(predictions_val, axis=1)
		return self.predicted_classes

	def print_results(self):
		right1 = 0
		false1 = 0
		right5 = 0
		false5 = 0
		right1 = 0
		for i in range(self.predicted_classes.shape[0]):
			if self.config["debug"]:
                		print("For image " + str(i+1) + " the top five are: ")
                		for j in range(5):
                        		print(" " + str(j+1) + ": " + self.mPred.getHumanLabel(self.predicted_classes[i][self.config["Model"][self.config["model"]]["num_clss"]-1-j]-1))	
			if self.mTru.isTop1(self.ground_truth[i]-1, self.mPred, self.predicted_classes[i]):
                		right1+=1
			else:
                		false1+=1
			if self.mTru.isTop5(self.ground_truth[i]-1, self.mPred, self.predicted_classes[i]):
                		right5+=1
			else:
                		false5+=1
		print("Mantissa Width: " + str(self.config["mantissa_width"]))
		print("Exponent Width: " + str(self.config["exponent_width"]))
		print("Weight Mantissa Width: " + str(self.config["weight"]["m_w"]))
		print("Weight Exponent Width: " + str(self.config["weight"]["e_w"]))
		print("Top 1: "+ str(right1) + " correct and " + str(false1) + " false")
		print("Top 5: "+ str(right5) + " correct and " + str(false5) + " false")

	def __get_sysnet(self):
		fil = open(self.config["gr_truth_dir"] + "ILSVRC2012_validation_ground_truth.txt", "r")
		self.ground_truth = np.array(fil.read().splitlines(), int)
		self.mPred = Label(self.config["lab_map"] + self.map_path)
		self.mTru = Label(self.config["lab_map"] + "ILSVRCMap.txt")
		
	def set_width(self, m_w=None, e_w=None, s_w=None, w_m_w=None, w_e_w =None, w_s_w=None):
		if(m_w is not None):
			self.config["mantissa_width"]        = m_w
		if(e_w is not None):
			self.config["exponent_width"]        = e_w
		if(s_w is not None):
			self.config["shared_exponent_width"] = s_w
		if(w_m_w is not None):
			self.config["weight"]["m_w"]         = w_m_w
		if(w_e_w is not None):
			self.config["weight"]["e_w"]         = w_e_w
		if(w_s_w is not None):
			self.config["weight"]["s_w"]         = w_s_w
		if(w_m_w is not None or w_e_w is not None or w_s_w is not None):
			self.path_to_ckpt = self.config["to_ckpt"] + self.config["Model"][self.model]["to_quant_w"] + "model" + str(self.config["weight"]["m_w"]) + str(self.config["weight"]["e_w"]) +".ckpt" 
			tf.reset_default_graph()
			self.__set_up_inception()
	
	def quant_weights(self, path_to_dir):
		tf.reset_default_graph()
		self.__set_up_inception()	
		saver = tf.train.Saver()
		np.set_printoptions(linewidth=np.inf)
		with tf.Session() as sess:
			self.init_assign_fn(sess)
			for i in range(len(self.variables_to_restore)):
				self.variables_to_restore[i].assign(bfp_out_module.bfp_out(self.variables_to_restore[i], ShDepth = self.config["weight"]["s_w"], MWidth=self.config["weight"]["m_w"], EWidth=self.config["weight"]["e_w"])).eval()
			save_path = saver.save(sess, path_to_dir+"model"+str(self.config["weight"]["m_w"])+str(self.config["weight"]["e_w"]) +".ckpt")
			print("Model saved in path: %s" %save_path) 
	
	def test_weights(self, path_to_dir):
		tf.reset_default_graph()
		self.__set_up_inception()
		self.assign_weights(path_to_dir)
		with tf.Session() as sess:
			self.init_assign_fn(sess)
			print("Weights: ")
			print(sess.run(self.variables_to_restore[0]))
		
	def assign_weights(self, path_to_dir):
		self.init_assign_fn = slim.assign_from_checkpoint_fn(path_to_dir, self.variables_to_restore)	


	def __read_yaml(self):
		stream = open(self.path_to_yaml, "r")
		self.config = yaml.load(stream)

	def __set_up(self):
		self.model = self.config["model"]
		if(self.model == "InceptionV1"):
			self.__set_up_inception()
			self.__set_up_images(self.config["Model"]["InceptionV1"]["preprocess"])	
		if(self.model == "VGG16"):
			self.__set_up_vgg16()
			self.__set_up_images(self.config["Model"]["VGG16"]["preprocess"])
		if(self.model == "VGG19"):
			self.__set_up_vgg19()
			self.__set_up_images(self.config["Model"]["VGG19"]["preprocess"])
		if(self.model == "ResNetV150"):
			self.__set_up_resnetv150()
			self.__set_up_images(self.config["Model"]["ResNetV150"]["preprocess"])

	def __set_up_inception(self):
		self.num_clss = self.config["Model"]["InceptionV1"]["num_clss"]
		self.map_path = self.config["Model"]["InceptionV1"]["map_path"]
		self.X = tf.placeholder(tf.float32, shape =[None, self.config["image"]["height"], self.config["image"]["width"], self.config["image"]["num_channels"]])
		with slim.arg_scope(inception.inception_v1_arg_scope()):
			self.logits, self.end_points = inception.inception_v1(self.X, num_classes=self.num_clss, is_training=False, m_w = self.config["mantissa_width"], e_w = self.config["exponent_width"], s_w = self.config["shared_exponent_width"], alter=self.config["quantize"], debug=self.config["debug_layer"])
		self.predictions = self.end_points["Predictions"]
		self.variables_to_restore = slim.get_variables_to_restore()
		self.init_assign_fn = slim.assign_from_checkpoint_fn(self.config["to_ckpt"]+self.config["Model"]["InceptionV1"]["ckpt_path"], self.variables_to_restore)
	
	def __set_up_vgg16(self):
		print("To be Implemented")

	def __set_up_vgg19(self):
		print("To be Implemented")
		
	def __set_up_resnetv150(self):
		print("To be Implemented")

	def __set_up_images(self, model):
		self.batch = np.empty([self.config["batch_size"], self.config["image"]["height"], self.config["image"]["width"], self.config["image"]["num_channels"]])
		for i in range(self.config["batch_size"]):
			path = self.config["data_dir"] + "ILSVRC2012_val_" + str(i+1).zfill(8) + ".jpeg"
			self.batch[i] = ImgProc(path, i+1).preprocess(arch=model)



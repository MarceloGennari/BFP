import yaml
import numpy as np
import tensorflow as tf
from nets import inception

slim = tf.contrib.slim

class BaseEngine:
  is_set = False

  def __init__(self):
    self._set_up_inception_()

  def _read_yaml_(self, path_to_yaml):
    stream = open(path_to_yaml, "r")
    return yaml.load(stream)

  def _assign_weights_(self, path_to_ckpt):
    self.init_assign_fn = slim.assign_from_checkpoint_fn(path_to_ckpt, self.variables_to_restore)

 # def __quant_arg_scope_():
	# Defines the scope of the bfp modules
	# To be Implemented

  def _set_up_inception_(self):
    if(BaseEngine.is_set):
      self.predictions = BaseEngine.predictions
      self.variables_to_restore = BaseEngine.variables_to_restore
      self._assign_weights_("/mnt/d/Data/Inception/inception_v1.ckpt")
      self.X = BaseEngine.X 
      return
    imcon = self._read_yaml_("../config/image_setup.yaml")
    self.X = tf.placeholder(tf.float32, shape =[None, imcon["height"], imcon["width"], imcon["nChan"]])
    with slim.arg_scope(inception.inception_v1_arg_scope()):
      _ , end_points = inception.inception_v1(self.X, num_classes=1001, is_training=False)
    self.predictions = end_points["Predictions"]
    self.variables_to_restore = slim.get_variables_to_restore()
    self._assign_weights_("/mnt/d/Data/Inception/inception_v1.ckpt")
    BaseEngine.X = self.X
    BaseEngine.predictions = self.predictions
    BaseEngine.variables_to_restore = self.variables_to_restore
    BaseEngine.is_set = True

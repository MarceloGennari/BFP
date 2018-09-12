import yaml
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/Archit/')

import inception

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

  def _reset_inception_(self, alter=False, m_w=20, e_w=8, arch='v1'):
    tf.reset_default_graph()
    self.__set_up_base__(alter=alter, m_w=m_w, e_w=e_w, arch=arch)


  def _set_up_inception_(self, alter=False, m_w=20, e_w=8, arch='v1'):
    if(BaseEngine.is_set):
      self.predictions = BaseEngine.predictions
      self.variables_to_restore = BaseEngine.variables_to_restore
      self._assign_weights_("/mnt/d/Data/Inception/inception_v1.ckpt")
      self.X = BaseEngine.X 
      return
    self.__set_up_base__(alter=alter, m_w=m_w,e_w=e_w, arch=arch)

  def __set_up_base__(self, alter=False, m_w=20, e_w=8, arch ='v1'):
    imcon = self._read_yaml_("../config/image_setup.yaml")
    self.X = tf.placeholder(tf.float32, shape =[None, imcon["height"], imcon["width"], imcon["nChan"]])
    scp, arch = self.__ret_incep_arch__(arch)
    with slim.arg_scope(scp()):
      _ , end_points= arch(self.X,num_classes=1001,is_training=False,alter=alter,m_w=m_w,e_w=e_w)
    self.predictions = end_points["Predictions"]
    self.variables_to_restore = slim.get_variables_to_restore()
    self._assign_weights_("/mnt/d/Data/Inception/inception_v1.ckpt")
    BaseEngine.X = self.X
    BaseEngine.predictions = self.predictions
    BaseEngine.variables_to_restore = self.variables_to_restore
    BaseEngine.is_set = True

  def __ret_incep_arch__(self, which='v1'):
    switcher={
	'v1': inception.inception_v1,
	'HW': inception.HWInception_v1}
    scp = {
	'v1': inception.inception_v1_arg_scope,
	'HW': inception.HWInception_v1_arg_scope}
    return scp[which], switcher[which]

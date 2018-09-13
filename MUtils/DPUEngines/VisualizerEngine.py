import BaseEngine
import tensorflow as tf
import numpy as np

class VisualizerEngine(BaseEngine.BaseEngine):
	
  def __init__(self, alter=False):
    self._set_up_inception_(alter=alter)

  def get_weights(self, path_to_ckpt, which=None):
    self._assign_weights_(path_to_ckpt)
    weights = np.array([])
    if which==None:
      rg = range(len(weights))
    else:
      rg = which
    with tf.Session() as sess:
      self.init_assign_fn(sess)
      for i in rg:
        weights = np.append(weights, self.variables_to_restore[i].eval())
    return weights

  def get_original_weights(self, which=None):
    weights = self.get_weights('/mnt/d/Data/Inception/inception_v1.ckpt', which)
    maximum = np.array([])
    if which==None:
      rg = range(len(weights))
    else:
      rg = which
    for i in rg:
      maxAbs = np.absolute(np.amax(weights[i]))
      minAbs = np.absolute(np.amin(weights[i]))
      absMax = maxAbs if maxAbs > minAbs else minAbs
      maximum = np.append(maximum, absMax) 
    return weights, maximum 

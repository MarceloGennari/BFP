import BaseEngine
import tensorflow as tf
import numpy as np
import math
class TransformEngine(BaseEngine.BaseEngine):

  def __init__(self, alter=False, arch='v1'):
    self._set_up_inception_(alter=alter, arch=arch)

  def transform_to_bias(self):
    """
	The whole point is to transform the checkpoint frm batch_norm to bias
        The slim 'v1' architecture do not use the gamma variable, so the equations are:
		weights = weights/moving_variance
		beta = beta-(moving_mean/moving_variance)
		moving_variance = 1
		moving_mean = 0
        This should provide the same output as using batchnorm
    """
    self._assign_weights_('/mnt/d/Data/Inception/inception_v1.ckpt')
    saver = tf.train.Saver() 
    # For the next 228 layers, we will get in groups of four, the weights, the beta, the moving_mean and mov_average
    # This means we have to iterate in 57 layers
    with tf.Session() as sess:
      self.init_assign_fn(sess)     
      for i in range(57): 
        weights = self.variables_to_restore[4*i].eval()
        beta = self.variables_to_restore[4*i+1].eval()
        moving_mean = self.variables_to_restore[4*i+2].eval()
        moving_variance = self.variables_to_restore[4*i+3].eval()
        numBatches = beta.size
        for j in range(numBatches):
          weights[:,:,:,j] = weights[:,:,:,j]/math.sqrt(moving_variance[j])
          beta[j] = beta[j]-(moving_mean[j]/math.sqrt(moving_variance[j]))
          moving_mean[j] = 0
          moving_variance[j] = 1
        self.variables_to_restore[4*i].assign(weights).eval()
        self.variables_to_restore[4*i+1].assign(beta).eval()
        self.variables_to_restore[4*i+2].assign(moving_mean).eval()
        self.variables_to_restore[4*i+3].assign(moving_variance).eval()
      save_path = saver.save(sess, '/mnt/d/Data/Inception/inception_v1_noBatch.ckpt')
      print("Model saved in path: %s" %save_path)

  def transform_to_INT_Bias(self):
    """
	The idea is to multiply all biases to be in the range for the input image (which was multiplied by 127)
    """
    self._assign_weights_('/mnt/d/Data/Inception/inception_v1_noBatch.ckpt')
    saver = tf.train.Saver()
    with tf.Session() as sess:
      self.init_assign_fn(sess)
      for i in range(57):
        beta = self.variables_to_restore[4*i + 1].eval()
        numBatches = beta.size
        for j in range(numBatches):
          beta[j] = 127*beta[j]
        self.variables_to_restore[4*i+1].assign(beta).eval()
      # This is the last bias of the fully connected layer
      beta = self.variables_to_restore[229].eval()
      self.variables_to_restore[229].assign(beta).eval()
      save_path = saver.save(sess, '/mnt/d/Data/Inception/inception_v1_noBatch_biasScaled.ckpt')
      
      print("Model saved in path: %s" %save_path)

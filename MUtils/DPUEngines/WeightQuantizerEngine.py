import BaseEngine
import tensorflow as tf
import numpy as np

class WeightQuantizerEngine(BaseEngine.BaseEngine):

  def __init__(self):
    self._set_up_inception_()
    quant_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/quant_out.so')
    self.quant = quant_out_module.quant_out

  def quant_weights(self, w_e_w, w_m_w, quantType, path_to_dir):
    saver = tf.train.Saver()
    self._assign_weights_('/mnt/d/Data/Inception/inception_v1.ckpt')
    with tf.Session() as sess:
      self.init_assign_fn(sess)
      for i in range(len(self.variables_to_restore)):
        if(self.variables_to_restore[i].name.split("/")[-1].split(":")[0]=="moving_variance"):
          continue
        res = self.variables_to_restore[i].eval()
        maxAbs = np.absolute(np.amax(res))
        minAbs = np.absolute(np.amin(res))
        absMax = maxAbs if maxAbs > minAbs else minAbs
        self.variables_to_restore[i].assign(self.quant( self.variables_to_restore[i],
							Scaling = absMax,
							MWidth = w_m_w,
							EWidth = w_e_w,
							FloatType=quantType)).eval()
      save_path = saver.save(sess,path_to_dir+"modelE"+str(w_e_w)+"M"+str(w_m_w)+".ckpt")
      print("Model saved in path: %s" %save_path)
    return save_path

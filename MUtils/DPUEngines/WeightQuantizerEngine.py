import BaseEngine
import tensorflow as tf
import numpy as np

class WeightQuantizerEngine(BaseEngine.BaseEngine):

  def __init__(self, alter = False, arch='v1'):
    self._set_up_inception_(alter=alter,arch=arch)
    quant_out_module = tf.load_op_library('/home/marcelo/tensorflow/Scripts/BFP/lib/quant_out.so')
    self.quant = quant_out_module.quant_out

#  def _4_dim_graph_(self, _4_dim_tensor):
    

  def _define_graph_(self, w_e_w, w_m_w, quantType):
    self.in_tens = tf.placeholder(tf.float32)
    # Receives all of the tensors and feature maps
    # For each self.variables_to_restore, we need to assign each of them
    # So if the tensor received is a 4-dimensional tensor, then break it up
    #condition = tf.shape(self.in_tens).size()
    #br_or_not = tf.cond(condition ==4, lambda:_4_dim_graph(), lambda: _l4_dim_graph())
 
    
    abs_in = tf.abs(self.in_tens)
    max_tens = tf.reduce_max(abs_in)
    quantized = self.quant(self.in_tens,max_tens,EWidth=w_e_w, MWidth=w_m_w, FloatType=quantType)
    return quantized

  def quant_weights(self, w_e_w, w_m_w, quantType, path_to_dir):
    saver = tf.train.Saver()
    self._assign_weights_('/mnt/d/Data/Inception/inception_v1.ckpt')
    quantized = self._define_graph_(w_e_w,w_m_w,quantType) 
    with tf.Session() as sess:
      self.init_assign_fn(sess)
      for i in range(len(self.variables_to_restore)):
         if(self.variables_to_restore[i].name.split("/")[-1].split(":")[0]=="moving_variance"):
           continue
         end = self.variables_to_restore[i].assign(quantized)
         sess.run(end, feed_dict={self.in_tens: self.variables_to_restore[i].eval()})
      save_path = saver.save(sess,path_to_dir+"modelE"+str(w_e_w)+"M"+str(w_m_w)+".ckpt")
      print("Model saved in path: %s" %save_path)
    return save_path
      


  def ___quant_weights(self, w_e_w, w_m_w, quantType, path_to_dir):
#    saver = tf.train.Saver()
#    self._assign_weights_('/mnt/d/Data/Inception/inception_v1.ckpt')
    
    # This placeholder is going to have the variables with 4 dimensions or more
#    update_placeholder = tf.placeholder(tf.float)
    
    # Define the Model before running the session
#    for i in range(len(self.variables_to_restore)):
#      if(self.variables_to_restore[i].name.split("/")[-1].split(":")[0]=="moving_variance"):
#        continue
#      sl = tf.slice(update_placeholder, [0,0,0,0], [sh[0], sh[1], sh[2], 1])

    return self.comput_graph(w_e_w, w_m_w, quantType, path_to_dir) 
'''
    with tf.Session() as sess:
      self.init_assign_fn(sess)
      for i in range(len(self.variables_to_restore)):

        # This quantizes the weight per tensor bases
        if(self.variables_to_restore[i].name.split("/")[-1].split(":")[0]=="moving_variance"):
          continue

        res = self.variables_to_restore[i].eval()

        # This is to quantize the weight per feature map
        if res.ndim == 4 : 
          sh = tf.shape(self.variables_to_restore[i])
          for j in range(sh[3].eval()):
            sl = tf.slice(self.variables_to_restore[i], [0, 0, 0, 0], [sh[0], sh[1], sh[2], 1]) 
            absMax = self.__find_max__(sl.eval())
            print(sl)
            sl = self.quant(sl, Scaling=absMax,MWidth=w_m_w,EWidth=w_e_w,FloatType=quantType)
            if(j==0): assT = sl
            else: assT = tf.concat([assT, sl], axis = 3)
          self.variables_to_restore[i].assign(assT).eval()
          print(assT)
          continue
        
        absMax = self.__find_max__(res)
        self.variables_to_restore[i].assign(self.quant( self.variables_to_restore[i],
							Scaling = absMax,
							MWidth = w_m_w,
							EWidth = w_e_w,
							FloatType=quantType)).eval()
      save_path = saver.save(sess,path_to_dir+"modelE"+str(w_e_w)+"M"+str(w_m_w)+".ckpt")
      print("Model saved in path: %s" %save_path)
    return save_path

  def __find_max__(self, arr):
    maxAbs = np.absolute(np.amax(arr))
    minAbs = np.absolute(np.amin(arr))
    absMax = maxAbs if maxAbs>minAbs else minAbs
    return absMax

'''

import BaseEngine
import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/Preprocessing')
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/')
from img_proc import ImgProc
from sysnet_labels import Label
import numpy as np
import tensorflow as tf

class InferenceEngine(BaseEngine.BaseEngine):
  def __init__(self, alter=False, m_w=20, e_w =8, arch='v1'):
    self._set_up_inception_(alter=alter,m_w=m_w, e_w=e_w, arch=arch)
    self._set_up_images_()
    self.__get_sysnet__()

  def __get_sysnet__(self):
    fil = open("/mnt/d/Data/ILSVRC2012/ILSVRC2012_validation_ground_truth.txt", "r")
    self.ground_truth = np.array(fil.read().splitlines(),int)
    self.mPred = Label("/mnt/d/Data/Inception/MappingSysnet.txt")
    self.mTru = Label("/mnt/d/Data/ILSVRCMap.txt") 

  def _set_up_images_(self):
    mcon = self._read_yaml_("../config/inference_setup.yaml")
    imcon = self._read_yaml_("../config/image_setup.yaml")
    self.batch =  np.empty([mcon["batch_size"], imcon["height"], imcon["width"], imcon["nChan"]])
    for i in range(mcon["batch_size"]):
      path = "/mnt/d/Data/ILSVRC2012/ILSVRC2012_img_val/ILSVRC2012_val_" + str(i+1).zfill(8)+".jpeg"
      self.batch[i] = ImgProc(path, i+1).preprocess()

  def inference(self):
    mcon = self._read_yaml_("../config/inference_setup.yaml")
    batch_s = int(mcon["batch_size"]/mcon["epochs"])
    self.predicted_classes = np.empty([mcon["batch_size"], 1001], int)
    for i in range(mcon["epochs"]):
      it = int(i*batch_s)
      with tf.Session() as sess:
        self.init_assign_fn(sess)
        self.predictions_val = self.predictions.eval(feed_dict={self.X: self.batch[it:it+batch_s]})
        self.predicted_classes[it:it+batch_s] = np.argsort(self.predictions_val, axis=1)
    return self.predicted_classes

  def print_results(self, w_e_w, w_m_w):
    right1, false1, right5, false5 = 0, 0, 0, 0
    mcon = self._read_yaml_("../config/inference_setup.yaml")
    for i in range(self.predicted_classes.shape[0]):
      if mcon["debug"]:
        print("For image " + str(i+1) + " the top five are: ")
        for j in range(5):
          print(" " + str(j+1) + ": " + self.mPred.getHumanLabel(self.predicted_classes[i][1001-1-j]-1))
          print(" " + str(j+1) + ": " + str(self.predictions_val[i][self.predicted_classes[i][1001-1-j]]-1)) 
      if self.mTru.isTop1(self.ground_truth[i]-1, self.mPred, self.predicted_classes[i]):
        right1+=1
      else:
        false1+=1
      if self.mTru.isTop5(self.ground_truth[i]-1, self.mPred, self.predicted_classes[i]):
        right5+=1
      else:
        false5+=1
    acc1 = (right1/(right1+false1))/0.698
    acc5 = (right5/(right5+false5))/0.896
    print("Weight Mantissa Width: " + str(w_m_w))
    print("Weight Exponent Width: " + str(w_e_w))
    print("Top 1: "+ str(right1) + " correct and " + str(false1) + " false (acc: " + str(acc1)+")")
    print("Top 5: "+ str(right5) + " correct and " + str(false5) + " false (acc: " + str(acc5)+")") 

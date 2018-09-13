# THIS SHOULD BE A SIMPLE INFERENCE TEST TO SEE SOME RESULTS OF RUNNING THE MODEL ON A CHECKPOINT

import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import InferenceEngine

inf = InferenceEngine.InferenceEngine()
inf._assign_weights_('/mnt/d/Data/Inception/inception_v1_noBatch.ckpt')
inf.inference()
inf.print_results(8,23)

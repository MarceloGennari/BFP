# THIS SHOULD BE A SIMPLE INFERENCE TEST TO SEE SOME RESULTS OF RUNNING THE MODEL ON A CHECKPOINT

import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import InferenceEngine

checkpoint_path = { 0: '/mnt/d/Data/Inception/inception_v1_noBatch_biasScaled.ckpt',
		    1: '/mnt/d/Data/Inception/inception_v1_noBatch.ckpt',
		    2: '/home/marcelo/tensorflow/Scripts/MUtils/to_hardware/hardware_variables.ckpt'}

archit = { 0: "v1",
	   1: "HW"}

inf = InferenceEngine.InferenceEngine(arch=archit[1])
inf._assign_weights_(checkpoint_path[2])
inf.inference()
inf.print_results(8,23)

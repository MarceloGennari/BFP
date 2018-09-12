import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import InferenceEngine
import WeightQuantizerEngine
#Quantizer = WeightQuantizerEngine.WeightQuantizerEngine

m=3
e=0

Inf = InferenceEngine.InferenceEngine
inf = Inf(alter=True, e_w=e, m_w=m, arch="HW")

#inf._assign_weights_(save_path)
for e in range(9):
	for m in range(11):
		inf._reset_inception_(alter=True, e_w=e, m_w=m, arch="HW")
		inf._assign_weights_('/mnt/d/Data/Inception/checkpoints/modelE'+str(e)+'M'+str(m)+'.ckpt')
		inf.inference()
		inf.print_results(e,m)

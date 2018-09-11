import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils/DPUEngines/')

import InferenceEngine
import WeightQuantizerEngine
Quantizer = WeightQuantizerEngine.WeightQuantizerEngine
Inf = InferenceEngine.InferenceEngine

f = Quantizer()
inf = Inf()

for e in range(9):
	for m in range(11):
		save_path = f.quant_weights(e, m,'FloatFixedPoint', '/mnt/d/Data/Inception/checkpoints/')
		inf._assign_weights_(save_path)
		inf.inference()
		inf.print_results(e,m)


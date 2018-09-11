import InferenceEngine
import WeightQuantizerEngine
Quantizer = WeightQuantizerEngine.WeightQuantizerEngine
Inf = InferenceEngine.InferenceEngine

f = Quantizer()
save_path = f.quant_weights(0,4,'FloatFixedPoint', '/mnt/d/Data/Inception/checkpoints/')
inf = Inf()
inf._assign_weights_(save_path)
inf.inference()
inf.print_results(0, 4)

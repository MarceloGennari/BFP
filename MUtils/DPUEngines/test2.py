import VisualizerEngine
import sys
sys.path.insert(0, '/home/marcelo/tensorflow/Scripts/MUtils')
import FloatGenerator as Fg
import matplotlib.pyplot as plt

VEng = VisualizerEngine.VisualizerEngine

v = VEng()
weights, maximum = v.get_original_weights()

for i in range(len(weights)):
  pWeights = gen_fixfl(8,0, maximum[i])
  fig = plt.figure(figsize=(10,6))
  plt.hist(weights[i], bins='auto')
  for xc in pWeights:
    line1 = plt.axvline(x=xc, color='r', alpha=1, linestyle='--', lw=1)
  plt.yscale('log')
  plt.show()

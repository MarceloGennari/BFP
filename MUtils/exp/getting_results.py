import numpy as np

dirPath = '/mnt/d/TestBench/TestBenchApp/MarceloProbes/'
with open(dirPath+'FC_output.txt') as f:
	tr_map = np.array(f.read().splitlines())
	tr_map = tr_map.astype(np.int)
#print(tr_map)
print(np.argsort(tr_map))

from InfEngine import InfEngine
import time

infer = InfEngine("/home/marcelo/tensorflow/Scripts/src/model_setup.yaml")

for w_m_w in range(3,9,4):
	for w_e_w in range(3, 8,5):
		for m_w in range(3,9,6):
			for e_w in range(3,8,6):
				start = time.time()
				infer.set_width(w_m_w=w_m_w, w_e_w=w_e_w, m_w = m_w, e_w= e_w)
				infer.assign_weights("/mnt/d/Data/Inception/checkpoints/model" + str(w_m_w) + str(w_e_w) + ".ckpt")
				infer.inference()
				print(time.time()-start)
				infer.print_results()

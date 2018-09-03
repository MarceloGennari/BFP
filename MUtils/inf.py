from InfEngine import InfEngine

infer = InfEngine("/home/marcelo/tensorflow/Scripts/src/model_setup.yaml")

for w_m_w in range(2,9):
	for w_e_w in range(2, 8):
		for m_w in range(2,9):
			for e_w in range(2,8):
				infer.set_width(w_m_w=w_m_w, w_e_w=w_e_w, m_w = m_w, e_w= e_w)
				infer.assign_weights("/mnt/d/Data/Inception/checkpoints/model" + str(w_m_w) + str(w_e_w) + ".ckpt")
				infer.inference()
				infer.print_results()

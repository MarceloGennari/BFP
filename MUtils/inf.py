from InfEngine import InfEngine

infer = InfEngine("/home/marcelo/tensorflow/Scripts/src/model_setup.yaml")

	infer.set_width(w_m_w=w_m_w, w_e_w=w_e_w, m_w = 4, e_w= 4)
	infer.assign_weights("/mnt/d/Data/Inception/checkpoints/model" + str(w_m_w) + str(w_e_w) + ".ckpt")
	infer.inference()
	infer.print_results()

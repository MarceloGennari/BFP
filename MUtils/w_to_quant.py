from InfEngine import InfEngine

infer = InfEngine("/home/marcelo/tensorflow/Scripts/src/model_setup.yaml")

for w_e_w in range(1, 9):
	for w_m_w in range(0, 11):
		infer.set_width(w_m_w=w_m_w, w_e_w=w_e_w)
		infer.quant_weights("/mnt/d/Data/Inception/checkpoints/")

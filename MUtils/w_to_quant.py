from InfEngine import InfEngine

w_m_w = 0
w_e_w = 4

infer = InfEngine("/home/marcelo/tensorflow/Scripts/src/model_setup.yaml")

infer.test_weights("/mnt/d/Data/Inception/inception_v1.ckpt")
infer.set_width(w_m_w=w_m_w, w_e_w=w_e_w)
infer.quant_weights("/mnt/d/Data/Inception/checkpoints/")


infer.test_weights("/mnt/d/Data/Inception/checkpoints/model" + str(w_m_w) + str(w_e_w) +  ".ckpt")
infer.assign_weights("/mnt/d/Data/Inception/checkpoints/model" + str(w_m_w) + str(w_e_w) + ".ckpt")
infer.inference()
infer.print_results()

'''
for w_e_w in range(1, 9):
	for w_m_w in range(0, 11):
		infer.set_width(w_m_w=w_m_w, w_e_w=w_e_w)
		infer.quant_weights("/mnt/d/Data/Inception/checkpoints/")
'''


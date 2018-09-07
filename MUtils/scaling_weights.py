from InfEngine import InfEngine

infer = InfEngine("/home/marcelo/tensorflow/Scripts/MUtils/config/model_setup.yaml")

infer.scale_weights(4, save=True)
#infer.inference()
#infer.print_results()
infer.assign_weights("/mnt/d/Data/Inception/Scaled/scaled_weight.ckpt")
infer.inference()
infer.print_results()

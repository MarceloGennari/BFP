from InfEngine import InfEngine

infer = InfEngine("/home/marcelo/tensorflow/Scripts/src/model_setup.yaml")
infer.test_weights("/mnt/d/Data/Inception/checkpoints/model108.ckpt")

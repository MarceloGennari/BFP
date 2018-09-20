#!/bin/bash
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
g++ -std=c++11 -I include/ -shared src/bfp_out.cc src/weight_quantizer.cc src/fp_types.cc -o lib/bfp_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -I include/ -shared src/quant_out.cc src/weight_quantizer.cc src/fp_types.cc -o lib/quant_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -I include/ -shared src/Int8_out.cc src/weight_quantizer.cc src/fp_types.cc -o lib/int8_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -I include/ -shared src/PerTensor_Int8.cc src/weight_quantizer.cc src/fp_types.cc -o lib/pertensor_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

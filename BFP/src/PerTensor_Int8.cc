#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantizer.h"
#include <assert.h>
#include <cmath>
#include <iostream>

using namespace tensorflow;

REGISTER_OP("PertensorOut")
        .Input("to_quant: float")
	.Output("quantized: float")
	.Output("scale: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	c->set_output(0, c->input(0));
	c->set_output(1, c->input(0));
	return Status::OK();
	});

class PertensorOutOp : public OpKernel{
	public:
		explicit PertensorOutOp(OpKernelConstruction* context) : OpKernel(context){
		}

		void Compute(OpKernelContext* context) override {
		/*
		* The idea is to quantize the weight / biases integrating the quantization and the shift
		* This function will quantize the weight based on the distance to a lookup table and will output:
		*	0-> The new values of the weights / biases
		*	1-> The shift (power of 2) needed to transform the weights / biases to INT8
		*/	
			DCHECK_EQ(1, context->num_inputs());
	
			// Grab the input tensor
			// Notice that those are all tensorflow::Tensor, not Eigen::Tensor
			const Tensor& input_tensor = context->input(0);
                        const TensorShape& shp = input_tensor.shape();
		
			// The idea now is to multiply input with a scaling factor that fits in INT8
			Tensor* output_tensor = NULL;
			Tensor* shift_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
			OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor.shape(), &shift_tensor));
			
			DCHECK_EQ(4, shp.dims());
			
			// This means that the inputs are the weights
			// The autos are Eigen Maps from Tensors
			auto shift_flat = shift_tensor->flat<float>();
			auto output_flat = output_tensor->flat<float>();	
			Eigen::Tensor<float, 1, 1> input_flat = input_tensor.flat<float>();	
			
			if(shp.dims()==4){
				Eigen::Tensor<float, 1, 1> abs_tens = input_flat.abs();
				Eigen::Tensor<float, 0, 1> maximus = abs_tens.maximum(Eigen::array<int, 1>({0}));
				float scaling = 127/maximus(0);				

				for(int k = 0; k<input_flat.size(); k++){
					output_flat(k) = input_flat(k)*scaling;
					shift_flat(k) = scaling;
				}
			}

		}

	private:
		WeightQuantizer q;	
};

REGISTER_KERNEL_BUILDER(Name("PertensorOut").Device(DEVICE_CPU), PertensorOutOp); 

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantizer.h"
#include <assert.h>
#include <cmath>
#include <iostream>

using namespace tensorflow;

REGISTER_OP("Int8Out")
        .Input("to_quant: float")
	.Output("quantized: float")
	.Output("shiftvalue: int32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	c->set_output(0, c->input(0));
	c->set_output(1, c->input(0));
	return Status::OK();
	});

class Int8OutOp : public OpKernel{
	public:
		explicit Int8OutOp(OpKernelConstruction* context) : OpKernel(context){
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

			auto shift_flat = shift_tensor->flat_inner_dims<int, 2>();
			auto output_flat = output_tensor->flat_inner_dims<float, 2>();			
			auto input_flat = input_tensor.flat_inner_dims<float, 2>();	

			DCHECK_EQ(4, shp.dims());

			if(shp.dims()==4){
				int d0 = shp.dim_size(0);
				int d1 = shp.dim_size(1);
				int d2 = shp.dim_size(2);
				int d3 = shp.dim_size(3);	
				
				Eigen::array<int, 3> dims({0,1,2 /*Reduce along the first three dimensions*/});
				// Now we need to transform from tensorflow::Tensor to Eigen::Tensor<dtype, numRows, numCols, numDepth>
				Eigen::Tensor<float, 4, 1> eigen_input_tensor = input_tensor.tensor<float, 4>();
				Eigen::Tensor<float, 4, 1> abs_tens = eigen_input_tensor.abs();
				Eigen::Tensor<float, 1, 1> maximus = abs_tens.maximum(Eigen::array<int,3>({0,1,2}));
				// After getting the maximum value of each Tensor, we have to map it to the nearest power of two				

				for(int k = 0; k<d3; k++){
					int scaling = (int) std::pow(2, std::ceil(std::log2(maximus(k))));
					int sc1 = (int) std::ceil(std::log2(maximus(k)));
					int sc2 = sc1-8;
					float scale = std::pow(2,sc1) - std::pow(2, sc2);
					q.set(0, 8, scale, DistType::FixedPoint);
					for(int j = 0; j< d0*d1*d2; j++){
						shift_flat(j,k) = -sc2;
						output_flat(j,k) = q.to_closest(input_flat(j,k));
					}
				}
			}

		}

	private:
		WeightQuantizer q;	
};

REGISTER_KERNEL_BUILDER(Name("Int8Out").Device(DEVICE_CPU), Int8OutOp); 

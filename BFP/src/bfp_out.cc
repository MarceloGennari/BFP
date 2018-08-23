#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <assert.h>
#include <cmath>

using namespace tensorflow;

REGISTER_OP("BfpOut")
	.Attr("ShDepth: int")
	.Attr("MWidth: int")
	.Attr("EWidth: int")
	.Input("to_bfp: float")
	.Output("bfped: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	c->set_output(0, c->input(0));
	return Status::OK();
	});

class BfpOutOp : public OpKernel{
	public:
		explicit BfpOutOp(OpKernelConstruction* context) : OpKernel(context){	
			// Grabbing the attributes for Shared Depth, Mantissa width and Exponent Width
			OP_REQUIRES_OK(context, context->GetAttr("ShDepth", &SharedDepth));
			OP_REQUIRES_OK(context, context->GetAttr("MWidth", &m_w));
			OP_REQUIRES_OK(context, context->GetAttr("EWidth", &e_w));

			// Checking if inputs are right
			OP_REQUIRES(context, SharedDepth>=1, errors::InvalidArgument("Need Shared Depth bigger than 0"));
			OP_REQUIRES(context, m_w >=0, errors::InvalidArgument("Need Mantissa Width bigger or equal to 0"));
			OP_REQUIRES(context, e_w >=1, errors::InvalidArgument("Need Exponent Width bigger than 0"));	
		}

		void Compute(OpKernelContext* context) override {
		
			// Grab the input tensor
			const Tensor& input_tensor = context->input(0);

			assert(input_tensor.dims() == 4);
			
			const TensorShape& shp = input_tensor.shape();
			
			// Get the dimensions of the Tensor
			const int batch = shp.dim_size(0);
			const int height = shp.dim_size(1);
			const int width = shp.dim_size(2);
			const int depth = shp.dim_size(3);		

			// Calculate how many Shared Exponent Blocks will be able to make it
			const int blocks = floor(depth/SharedDepth);
			const int last_block = depth-blocks*SharedDepth;		
		
			// Create an output tensor
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

			// Notice that the auto here expects type definition in the argument
			auto input = input_tensor.shaped<float, 4>({batch, height, width, depth});
			auto output = output_tensor->shaped<float, 4> ({batch, height, width, depth});			
			

			// NAIVE APPROACH TO QUANTIZE VALUES IN THE BLOCKS OF SHARED DEPTH 
			for (int i = 0; i<batch; i++){
				for( int h =0; h<height; h++){
					for(int w = 0; w< width; w++){
						for(int b = 0; b < blocks; b++){
							// This iterates through each block
							float highestValue;
							for(int d = b; d <(b+1)*SharedDepth; d++){
								if(d==b) highestValue = input(i,h,w,d); // Initialize highest value
								// This iterates through the values in each block
								if(input(i,h,w,d)>highestValue){
									highestValue = input(i,h,w,d);
								}
							}
							for(int d = b; d<(b+1)*SharedDepth; d++){
								output(i,h,w,d) = Quantize2(input(i,h,w,d));
							}
						}
					
						// This iterates through the last values which could not fit in one block
						float highestValue;
						for(int d = blocks*SharedDepth; d<depth; d++){
							if(d==blocks*SharedDepth) highestValue = input(i,h,w,d); // Initialize 
							if(input(i,h,w,d) > highestValue){
								highestValue = input(i,h,w,d);
							}
						}
						for(int d = blocks*SharedDepth; d<depth; d++){
							output(i,h,w,d) = Quantize2(input(i,h,w,d));
						}
					}
				}
			}
		
		}
		

	private:
		int SharedDepth, e_w, m_w;
	
		float Quantize1(float highestValue){
			// This quantization method simply sets everything the same for debugging
			return highestValue;
		}

		float Quantize2(float value){
			/***
			* Simple Implementation of Quantization:
			* For this quantization, it is assumed that values are in FP32
			* They will be transformed to 1-sign, 8-exponent, variable mantissa
			* This way, no scaling will be needed
			* The proposed method works as follows (since in c++ we can't really do bitwise operation)
			* 	1. if value is less than zero, multiply by -1.0f to get the absolute value
			*	2. take the log2 of the absolute value and floor it - this is the value of the exponent
			*	3. divide the absolute value by the value of 2^exponent, to get the value of 1.b0b1b2...
			*	4. multiply by 2^mantissa_bits to get 1b0b1b2b3.b4b5b6b7 etc
			*	5. floor the value to get 1b0b1b2b3.0000 etc
			*	6. divide by 2^mantissa_bits to get 1.b0b1b2b3
			*	7. multiply by the value 2^exponent
			* 	8. if it was less then zero, multiply by -1.0f to get the right value
			*	9. return the right value
			***/
			bool neg = false;
			if(value<0.0f){
				neg = true;
				value*=-1.0f;
			}
			int exp = floor(log2f(value));
			value /= pow(2, exp);
			// Now value should be 1.b0b1b2b3...
			value*=pow(2,m_w);
			value = floor(value);
			value  /= pow(2, m_w);
			value *=pow(2, exp);
			if(neg) value*=-1.0f;
			
			return value;
		}
};

REGISTER_KERNEL_BUILDER(Name("BfpOut").Device(DEVICE_CPU), BfpOutOp);















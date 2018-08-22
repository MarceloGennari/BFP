#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <assert.h>

using namespace tensorflow;

REGISTER_OP("BfpOut")
	.Input("to_bfp: float")
	.Output("bfped: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	c->set_output(0, c->input(0));
	return Status::OK();
	});

class BfpOutOp : public OpKernel{
	public:
		explicit BfpOutOp(OpKernelConstruction* context) : OpKernel(context), SharedDepth(16), m_w(5), e_w(5) {}

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
	

			// Create an output tensor
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

			// Notice that the auto here expects type definition in the argument
			auto input = input_tensor.shaped<float, 4>({batch, height, width, depth});
			auto output = output_tensor->shaped<float, 4> ({batch, height, width, depth});			
			// Set all but the first element of the output tensor to 0
			for (int i = 0; i<batch; i++){
				for( int h =0; h<height; h++){
					for(int w = 0; w< width; w++){
						for(int d = 0; d <depth; d++){
							output(i, h, w, d) = input(i, h, w, d);
							//else output(i,h,w,d) = 3;
						}
					}
				}
			}
		
			std::cout << "Height: " << height << " " << "Width: " << width << " " << "Depth: " << depth << std::endl;	

			// Preserve the first input value if possible
			//if(N>0) output_flat(0) = input(0);
				
			/****
				The Idea is:
					Find maximum exponent in depth of the input tensor
					Shift everything to match that exponent
					Return new tensor 
			****/
			//assert(dims[0] > 0);
			//assert(dims[1] > 0);
			//assert(dims[2] > 0);
		}
		

	private:
		int SharedDepth, m_w, e_w;
		int mantissa, exponent, sig;

		//void Quantize(
};

REGISTER_KERNEL_BUILDER(Name("BfpOut").Device(DEVICE_CPU), BfpOutOp);

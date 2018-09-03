#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantizer.h"
#include <assert.h>
#include <cmath>
#include <iostream>
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
			OP_REQUIRES(context, e_w >=0, errors::InvalidArgument("Need Exponent Width bigger or equal to 0"));	
			
			// Initializing the quantizer
			q.set(SharedDepth, e_w, m_w);	
		}

		void Compute(OpKernelContext* context) override {
		
			// Grab the input tensor
			const Tensor& input_tensor = context->input(0);

			//assert(input_tensor.dims() == 4);
			
			const TensorShape& shp = input_tensor.shape();
			
			// Get the dimensions of the Tensor
			//const int batch = shp.dim_size(0);
			//const int height = shp.dim_size(1);
			//const int width = shp.dim_size(2);
			//const int depth = shp.dim_size(3);		

			// Calculate how many Shared Exponent Blocks will be able to make it
			//const int blocks = floor(depth/SharedDepth);
			//const int last_block = depth-blocks*SharedDepth;		
		
			// Create an output tensor
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

			// Notice that the auto here expects type definition in the argument
			//auto input = input_tensor.shaped<float, 4>({batch, height, width, depth});
			//auto output = output_tensor->shaped<float, 4> ({batch, height, width, depth});			
			
			auto input_flat = input_tensor.flat<float>();
			auto output_flat = output_tensor->flat<float>();	
			for ( int k = 0; k<input_flat.size(); k++){
				output_flat(k) = q.to_var_fp(input_flat(k));
			}
			//std::cout << "Number of Crops: " << q.getNbCrop() << std::endl;
		
		}
		

	private:
		Quantizer q;
		int SharedDepth, e_w, m_w;
};

REGISTER_KERNEL_BUILDER(Name("BfpOut").Device(DEVICE_CPU), BfpOutOp);


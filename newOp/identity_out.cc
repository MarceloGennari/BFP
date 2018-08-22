#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;

REGISTER_OP("IdentityOut")
	.Input("to_id: float")
	.Output("identi: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	c->set_output(0, c->input(0));
	return Status::OK();
	});

class IdentOutOp : public OpKernel{
	public:
		explicit IdentOutOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {

			const Tensor& input_tensor = context->input(0);
			auto input = input_tensor.flat<float>();
			
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
										&output_tensor));
			auto output_flat = output_tensor->flat<float>();
			output_flat = input;
			
			std::cout << "This is printing" << std::endl;
		}
};

REGISTER_KERNEL_BUILDER(Name("IdentityOut").Device(DEVICE_CPU), IdentOutOp);

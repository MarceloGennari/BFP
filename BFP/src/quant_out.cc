#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantizer.h"
#include <assert.h>
#include <cmath>
#include <iostream>
using namespace tensorflow;

REGISTER_OP("QuantOut")
	.Attr("MWidth: int")
	.Attr("EWidth: int")
	.Attr("Scaling: float")
	.Attr("FloatType: {'FloatingPoint', 'FloatFixedPoint', 'FixedPoint'}")
	.Input("to_quant: float")
	.Output("quantized: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	c->set_output(0, c->input(0));
	return Status::OK();
	});

class QuantOutOp : public OpKernel{
	public:
		explicit QuantOutOp(OpKernelConstruction* context) : OpKernel(context){
			std::string tp;

			OP_REQUIRES_OK(context, context->GetAttr("MWidth", &m_w));
			OP_REQUIRES_OK(context, context->GetAttr("EWidth", &e_w));
			OP_REQUIRES_OK(context, context->GetAttr("Scaling", &sc));
			OP_REQUIRES_OK(context, context->GetAttr("FloatType", &tp));

			if(tp=="FloatingPoint") d = DistType::FloatingPoint;
			if(tp=="FloatFixedPoint") d = DistType::FloatFixedPoint;
			if(tp=="FixedPoint") d = DistType::FixedPoint;

			OP_REQUIRES(context, m_w >=0, errors::InvalidArgument("Need higher than zero mantissa"));
			OP_REQUIRES(context, e_w >=0, errors::InvalidArgument("Need higher than zero exponent"));
			
			
			// Initializing the quantizer
			q.set(e_w, m_w, sc, d);	
		}

		void Compute(OpKernelContext* context) override {
		
			// Grab the input tensor
			const Tensor& input_tensor = context->input(0);
			const TensorShape& shp = input_tensor.shape();
		
			// Create an output tensor
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
			
			auto input_flat = input_tensor.flat<float>();
			auto output_flat = output_tensor->flat<float>();	
			for ( int k = 0; k<input_flat.size(); k++){
				output_flat(k) = q.to_closest(input_flat(k));
			}
		}

	private:
		int m_w, e_w;
		float sc;
		DistType d;
		WeightQuantizer q;	
};

REGISTER_KERNEL_BUILDER(Name("QuantOut").Device(DEVICE_CPU), QuantOutOp); 

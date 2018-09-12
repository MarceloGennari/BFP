#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantizer.h"
#include <assert.h>
#include <cmath>
#include <iostream>
using namespace tensorflow;

REGISTER_OP("INT8Out")
	.Attr("MWidth: int")
	.Attr("EWidth: int")
	.Attr("FloatType: {'FloatingPoint', 'FloatFixedPoint', 'FixedPoint'}")
        .Input("to_quant: float")
	.Input("scaling: float")
	.Output("quantized: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
	c->set_output(0, c->input(0));
	return Status::OK();
	});

class INT8OutOp : public OpKernel{
	public:
		explicit QuantOutOp(OpKernelConstruction* context) : OpKernel(context){
			std::string tp;

			OP_REQUIRES_OK(context, context->GetAttr("MWidth", &m_w));
			OP_REQUIRES_OK(context, context->GetAttr("EWidth", &e_w));
			OP_REQUIRES_OK(context, context->GetAttr("FloatType", &tp));

			if(tp=="FloatingPoint") d = DistType::FloatingPoint;
			if(tp=="FloatFixedPoint") d = DistType::FloatFixedPoint;
			if(tp=="FixedPoint") d = DistType::FixedPoint;

			OP_REQUIRES(context, m_w >=0, errors::InvalidArgument("Need higher than zero mantissa"));
			OP_REQUIRES(context, e_w >=0, errors::InvalidArgument("Need higher than zero exponent"));
	
		}

		void Compute(OpKernelContext* context) override {
	
			DCHECK_EQ(2, context->num_inputs());
	
			// Grab the input tensor
			// Notice that those are all tensorflow::Tensor, not Eigen::Tensor
			const Tensor& input_tensor = context->input(0);
                        const Tensor& scaling_in = context->input(1);
                        const TensorShape& shp = input_tensor.shape();
			const TensorShape& shp2 = scaling_in.shape();
		
			// The idea now is to choose whether I want to quantize per feature map or per tensor.
			// The Tensor will be 4-dimensional, which will include many 3-dimensional kernels (feature maps)
			// Following the "Reduction Dimensions" section of this tutorial:
			// https://bitbucket.org/eigen/eigen/src/677c9f1577810e869f4f09881cabc3e503a810c1/unsupported/Eigen/CXX11/src/Tensor/README.md?at=default&fileviewer=file-view-default#markdown-header-reduction-operations
			// Create an output tensor
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
			
			auto input_flat = input_tensor.flat<float>();
			auto output_flat = output_tensor->flat<float>();	
			auto output_flat2 = output_tensor->flat_inner_dims<float, 2>();			
			auto input_flat2 = input_tensor.flat_inner_dims<float, 2>();	

			if(shp.dims()==4){
				int d0 = shp.dim_size(0);
				int d1 = shp.dim_size(1);
				int d2 = shp.dim_size(2);
				int d3 = shp.dim_size(3);	
				
				Eigen::array<int, 3> dims({0,1,2 /*Reduce along the first three dimensions*/});
				// Now we need to transform from tensorflow::Tensor to Eigen::Tensor
				Eigen::Tensor<float, 4, 1> eigen_input_tensor = input_tensor.tensor<float, 4>();
				Eigen::Tensor<float, 4, 1> abs_tens = eigen_input_tensor.abs();
				Eigen::Tensor<float, 1, 1> maximus = abs_tens.maximum(Eigen::array<int,3>({0,1,2}));

				for(int k = 0; k<d3; k++){
					q.set(e_w, m_w, maximus(k), d);
					for(int j = 0; j< d0*d1*d2; j++){
						output_flat2(j,k) = q.to_closest(input_flat2(j,k));
					}
				}
				return;
			}


			// This guarantees that Input2 is a scalar
			DCHECK_EQ(shp2.dims(), 0);
			
			auto scaling = scaling_in.flat<float>();
			sc = scaling(0);
			
			// Initializing the quantizer
			q.set(e_w, m_w, sc, d);	
	
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

REGISTER_KERNEL_BUILDER(Name("INT8Out").Device(DEVICE_CPU), INT8OutOp); 

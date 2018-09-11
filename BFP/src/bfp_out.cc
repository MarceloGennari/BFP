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
	.Attr("offset: int")
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
			//q.set(SharedDepth, e_w, m_w, ofs);	
		}

		void Compute(OpKernelContext* context) override {
		
			// Grab the input tensor
			const Tensor& input_tensor = context->input(0);

			const TensorShape& shp = input_tensor.shape();
		
			// Create an output tensor
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

			auto input_flat = input_tensor.flat_inner_dims<float,2>();
			auto output_flat=output_tensor->flat_inner_dims<float,2>();

			Eigen::Tensor<float, 1, 1> maximus;
			int perFm = 0;
			int nbFm = 0;
			
			OP_REQUIRES(context, shp.dims()==4, errors::InvalidArgument("Need dimension 4"));	

			// The input can be either 3 or 4 dimensional
			if(shp.dims() ==4){
				int d0 = shp.dim_size(0);
				int d1 = shp.dim_size(1);
				int d2 = shp.dim_size(2);
				int d3 = shp.dim_size(3);
				Eigen::Tensor<float, 4, 1> eigen_input_tensor = input_tensor.tensor<float, 4>();
				Eigen::Tensor<float, 4, 1> abs_tens = eigen_input_tensor.abs();
				maximus = abs_tens.maximum(Eigen::array<int,3>({0,1,2}));
				perFm = d0*d1*d2;
				nbFm = d3;
			}	

			/*
			if(shp.dims() == 3){
				int d0 = shp.dim_size(0);
				int d1 = shp.dim_size(1);
				int d2 = shp.dim_size(2);
				Eigen::Tensor<float, 3, 1> eigen_input_tensor = input_tensor.tensor<float, 3>();
				Eigen::Tensor<float, 3, 1> abs_tens = eigen_input_tensor.abs();
				maximus = abs_tens.maximum(Eigen::array<int,2>({0,1}));
				perFm = d0*d1;
				nbFm = d2;
			}
			*/

			for(int k = 0; k<nbFm; k++){
				q.set(e_w, m_w, maximus(k));
				for(int j = 0; j<perFm; j++){
					output_flat(j,k) = q.to_closest(input_flat(j, k));
				}
			}	
		}
		

	private:
		WeightQuantizer q;
		int SharedDepth, e_w, m_w, ofs;
};

REGISTER_KERNEL_BUILDER(Name("BfpOut").Device(DEVICE_CPU), BfpOutOp);

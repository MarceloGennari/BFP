#ifndef GUARD_QUANT
#define GUARD_QUANT

#include <cmath>
#include <assert.h>
#include <stdint.h>
#include <cstring>
#include <vector>
#include "fp_types.h"

template<typename T>
class QuantizerBase{
	public:
		T to_closest(T value);
		void print_dist();		
		virtual ~QuantizerBase() = 0;
	
	protected:
		int e_w, m_w;
		int bitSize;
		std::vector<T> fp;

		bool is_set_fp();
		void set_bitSize(int bitSize);	
		void set(int EWidth, int MWidth, DistType d);
		void set(int EWidth, int MWIdth);	
		void set_e_w(int EWidth);
		void set_m_w(int MWidth);
		void set_fp(DistType d);
};


#include "quantizer_base.inl"

class Quantizer : public QuantizerBase<float>{
	public:
		Quantizer();
		Quantizer(int sh, int e_w, int m_w, int ofs, DistType = DistType::FloatFixedPoint);
		void set(int sh, int e_w, int m_w, int ofs, DistType = DistType::FloatFixedPoint);
	
		float to_var_fp(float value);
		float to_var_fp_arith(float value);	

		static int nb_crop_up;
		static int nb_crop_down;

	private:
		int SharedDepth, exp_offset;
		int32_t MAX_EXP, MIN_EXP;
};

class WeightQuantizer : public QuantizerBase<float> {
	public:
		WeightQuantizer();	
		WeightQuantizer(int EWidth, int MWidth, float sc, DistType = DistType::FloatFixedPoint);
		void set(int EWidth, int MWidth, float sc, DistType = DistType::FloatFixedPoint);

	private:
		float sc;
};

class HardwareQuantizer : public QuantizerBase<int> {
	public:
		HardwareQuantizer();
		HardwareQuantizer(int bitSize, DistType = DistType::INT);
		void set(int bitSize, DistType = DistType::INT);

	private:
		int bitSize;
};

#endif

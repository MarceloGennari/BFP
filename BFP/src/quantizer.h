#ifndef GUARD_QUANT
#define GUARD_QUANT

#include <cmath>
#include <assert.h>
#include <stdint.h>
#include <cstring>

class Quantizer{
	public:
		Quantizer();
		Quantizer(int sh, int e_w, int m_w);
		void set(int sh, int e_w, int m_w);
	
		float to_var_fp(float value);

		float to_var_fp_arith(float value);	

		static int nb_crop_up;
		static int nb_crop_down;
	private:
		int SharedDepth, e_w, m_w;
		int exp_offset;
		int32_t MAX_EXP, MIN_EXP;
};

#endif

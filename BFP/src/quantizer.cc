#include "quantizer.h"
#include "bintypes.h"
#include <iostream>

int Quantizer::nb_crop_down = 0;
int Quantizer::nb_crop_up = 0;

// DistType is defaulted as FloatFixed point
void Quantizer::set(int sh, int e_w, int m_w, int ofs, DistType dist){
	QuantizerBase::set(e_w, m_w, dist);
	this->SharedDepth = sh;
	this->exp_offset = ofs;
	MAX_EXP = (1<<(e_w-1))-exp_offset;
	MIN_EXP = (-1*(1<<(e_w-1)))+1-exp_offset;
}

Quantizer::Quantizer(){}

// DistType is defaulted as FloatFixed point
Quantizer::Quantizer(int sh, int e_w, int m_w, int ofs, DistType dist){
	this->set(sh, e_w, m_w, ofs, dist);
}

float Quantizer::to_var_fp_arith(float v){
	/***
        * Simple Implementation of Quantization:
        * For this quantization, it is assumed that values are in FP32
        * They will be transformed to 1-sign, variable exponent, variable mantissa
        * This strategy adopts the truncation of the exponent
        * So given the exponent width e_w, the exponent will have to be between:
        *                       1-pow(2,e_w-1) <= exponent <= pow(2,e_w-1)
        * Any values that are above or below those extremes will be truncated to the max/min value
        * The proposed method works as follows (since in c++ we can't really do bitwise operation)
        *       1. if value is less than zero, multiply by -1.0f to get the absolute value
        *       2. take the log2 of the absolute value and floor it - this is the value of the exponent
        *       3. divide the absolute value by the value of 2^exponent, to get the value of 1.b0b1b2...
        *       4. multiply by 2^mantissa_bits to get 1b0b1b2b3.b4b5b6b7 etc
        *       5. floor the value to get 1b0b1b2b3.0000 etc
        *       6. divide by 2^mantissa_bits to get 1.b0b1b2b3
        *       7. check if exponent is between extreme values
        *       8. if it isn't, truncate it to the maximum value
        *       9. multiply by the value 2^exponent
        *       10. if it was less then zero, multiply by -1.0f to get the right value
        *       11. return the right value
        ***/
	bool neg = false;
	if(v < 0.0f){
		neg = true;
		v*=-1.0f;
	}
	if(v == 0.0f){
		return v;
	}
	int exp = floor(log2f(v));
	v /= pow(2, exp);
	// Now value should be 1.b0b1b2b3...
	v*= pow(2,m_w);
	v = floor(v);
	v/=pow(2,m_w);
	// Now check if exp is in desired range and truncate it if not
	int tmp = pow(2, e_w-1);
	if(exp<1-tmp) exp = 1-tmp;
	if(exp > tmp) exp = tmp;
	v*=pow(2, exp);
	if(neg) v*=-1.0f;

	return v;
}

float Quantizer::to_var_fp(float v){
	/***
	* This implementation should run very fast since it just uses shifting and bitwise logic operators 
	* Remember that the assumption is that float is FP32 with (from MSB), 1bit sign, 8 bit exp, 23 bit mantissa
	***/
	if(v == 0){
		return v;
	}
	uint32_t i;
	assert(sizeof(v) == sizeof(i));
	std::memcpy(&i, &v, sizeof(i));
	int32_t _exp = i & BF::EXP;
	_exp = _exp>>23;
	_exp = _exp-127;
	uint32_t _mant = i & BF::MANT;
	uint32_t _sign = i & BF::SIGN;
		
	if(_exp > MAX_EXP){
		_exp = MAX_EXP;
		_exp = _exp+127;
		_exp = _exp<<23;
		_mant = BF::MANT & BF::MASKM[m_w];
		i = _mant + _sign + _exp;
		std::memcpy(&v, &i, sizeof(i));
		nb_crop_up++;
		return v;
	}

	if(_exp < MIN_EXP){
		_exp = MIN_EXP;
		nb_crop_down++;
	}

	_exp = _exp + 127;
	_exp = _exp<<23;
	
	_mant = _mant & BF::MASKM[m_w];

	i = _mant + _sign + _exp;
	std::memcpy(&v, &i, sizeof(i));
	
	return v;
}

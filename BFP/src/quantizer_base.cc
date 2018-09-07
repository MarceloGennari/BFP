#include "quantizer.h"
#include <iostream>

float QuantizerBase::to_closest(float value){
        /***
        * This function returns the closest number to value from the vector fp
        * A note is that this will bring values to zero as well, if zero is available in the vector
        ***/
        if(value == 0) { return 0;}
        auto it = std::lower_bound(fp.begin(), fp.end(), value);
        if(it==fp.end()){it--; return *it;}
        if(it==fp.begin()){ return *it;}
        float diff_this = std::abs(*it-value);
        float diff_prev = std::abs(*(it-1)-value);
        if(diff_this>diff_prev) it--;

        return *it;	
}

bool QuantizerBase::is_set_fp(){
	bool is;
	if(fp.empty()) is=true;
	else is =false;
	return is;
}

void QuantizerBase::set(int EWidth, int MWidth){
	set_e_w(EWidth);
	set_m_w(MWidth);
}

void QuantizerBase::set(int EWidth, int MWidth, DistType d){
	set_e_w(EWidth);
	set_m_w(MWidth);
	set_fp(d);
}

void QuantizerBase::set_e_w(int EWidth){
	this->e_w = EWidth;
}

void QuantizerBase::set_m_w(int MWidth){
	this->m_w = MWidth;
}

void QuantizerBase::set_fp(DistType d){
	this->fp = FloatTypes::getDistribution(d, e_w, m_w);
}

void QuantizerBase::print_dist(){
	if(fp.empty()){ std::cout << "FP is Empty. Abort." << std::endl; return;}
	std::cout << "There are " << fp.size() << " elements in this distribution:" << std::endl;
	for(auto it=fp.begin(); it!=fp.end(); it++){
		std::cout << *it << " ";
	}
	std::cout << std::endl;
}

QuantizerBase::~QuantizerBase(){};

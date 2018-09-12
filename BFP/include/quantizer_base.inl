#include "quantizer.h"
#include <iostream>
#include <typeinfo>

template<class T>
inline T QuantizerBase<T>::to_closest(T value){
        /***
        * This function returns the closest number to value from the vector fp
        * A note is that this will bring values to zero as well, if zero is available in the vector
        ***/
        if(value == 0) { return 0;}
        auto it = std::lower_bound(fp.begin(), fp.end(), value);
        if(it==fp.end()){it--; return *it;}
        if(it==fp.begin()){ return *it;}
        T diff_this = std::abs(*it-value);
        T diff_prev = std::abs(*(it-1)-value);
        if(diff_this>diff_prev) it--;

        return *it;	
}

template<class T>
inline bool QuantizerBase<T>::is_set_fp(){
	bool is;
	if(fp.empty()) is=true;
	else is =false;
	return is;
}

template<class T>
inline void QuantizerBase<T>::set(int EWidth, int MWidth){
	set_e_w(EWidth);
	set_m_w(MWidth);
}

template<class T>
inline void QuantizerBase<T>::set(int EWidth, int MWidth, DistType d){
	set_e_w(EWidth);
	set_m_w(MWidth);
	set_fp(d);
}

template<class T>
inline void QuantizerBase<T>::set_e_w(int EWidth){
	this->e_w = EWidth;
}

template<class T>
inline void QuantizerBase<T>::set_m_w(int MWidth){
	this->m_w = MWidth;
}

template<class T>
inline void QuantizerBase<T>::set_bitSize(int bitSize){
	this->bitSize = bitSize;
}

template<class T>
inline void QuantizerBase<T>::set_fp(DistType d){
	this->fp = TypesWrapper<T>::getDistribution(d, e_w, m_w, bitSize);
}

template<class T>
inline void QuantizerBase<T>::print_dist(){
	if(fp.empty()){ std::cout << "FP is Empty. Abort." << std::endl; return;}
	std::cout << "There are " << fp.size() << " elements in this distribution:" << std::endl;
	for(auto it=fp.begin(); it!=fp.end(); it++){
		std::cout << *it << " ";
	}
	std::cout << std::endl;
}

template<class T>
inline QuantizerBase<T>::~QuantizerBase<T>(){};


#include "quantizer.h"
#include <iostream>

WeightQuantizer::WeightQuantizer(){};

// DistType is defaulted as FixedFloat in header file
WeightQuantizer::WeightQuantizer(int EWidth, int MWidth, float sc, DistType dist){
	this->set(EWidth, MWidth, sc, dist);	
}

// DistType is defaulted as FixedFloat in header file
void WeightQuantizer::set(int EWidth, int MWidth, float sc, DistType dist){
	this->QuantizerBase::set(EWidth, MWidth, dist);
	this->sc = sc;
	auto maxi = std::max_element(std::begin(fp), std::end(fp));

	/* This will keep the range of the weight to include the maximum value */
	for(auto it = fp.begin(); it!=fp.end(); it++){
		*it = *it*(sc/(*maxi));
	}
}

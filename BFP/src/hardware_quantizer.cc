#include "quantizer.h"
#include <iostream>

HardwareQuantizer::HardwareQuantizer(){};

HardwareQuantizer::HardwareQuantizer(int bitSize, DistType dist){
	this->set(bitSize, dist);
}

void HardwareQuantizer::set(int bitSize, DistType dist){
	this->QuantizerBase<int>::set_bitSize(bitSize);
	this->set_fp(dist);	
}

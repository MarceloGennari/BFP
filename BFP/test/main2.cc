#include "fp_types.h"
#include <iostream>

void print_values(std::vector<float> f){
	std::cout << "Size: " << f.size() << std::endl;
	for(auto it = f.begin(); it!=f.end(); it++){
		std::cout << *it << " ";
	}
	std::cout << std::endl;
}


int main(){
	std::vector<float> floatP = FloatTypes::getDistribution(DistType::FloatingPoint, 1, 2);
	std::vector<float> fixFloatP = FloatTypes::getDistribution(DistType::FloatFixedPoint, 1, 2);
	std::vector<float> fixP = FloatTypes::getDistribution(DistType::FixedPoint, 1, 2);
	std::vector<float> fixP2 = FloatTypes::getDistribution(DistType::FixedPoint, 3);
	std::vector<float> fixFloatP2 = FloatTypes::getDistribution(DistType::FloatFixedPoint, 0, 3);
	std::vector<float> ExOnly1 = FloatTypes::getDistribution(DistType::FloatFixedPoint, 3, 0);
	std::vector<float> ExOnly2 = FloatTypes::getDistribution(DistType::FloatingPoint, 3, 0);
	
	std::cout << "Floating Point: " << std::endl;
	print_values(floatP);
	std::cout << std::endl << "Fix Floating Point: " << std::endl;
	print_values(fixFloatP);
	std::cout << std::endl << "Fixed Point 1: " << std::endl;
	print_values(fixP); 	
	std::cout << std::endl << "Fixed Point 2: " << std::endl;
	print_values(fixP2);
	std::cout << std::endl << "Fixed Point using FixFloating Point: " << std::endl;
	print_values(fixFloatP2);
	std::cout << std::endl << "Exponent Only using FixFloating Point: " << std::endl;
	print_values(ExOnly1);
	std::cout << std::endl << "Exponent Only using Floating Point: " << std::endl;
	print_values(ExOnly2);
}

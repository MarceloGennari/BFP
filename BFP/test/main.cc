#include "quantizer.h"

int main(){
	std::vector<float> values = FloatTypes::getDistribution(DistType::FixedPoint, 3);
	for(auto it = values.begin(); it!= values.end(); it++){
		std::cout << *it/0.875 << std::endl;
	}	

	WeightQuantizer wq(1, 2, 3.5);
	wq.print_dist();
	std::cout << std::endl;
	WeightQuantizer wq2(0, 3, 7, DistType::FixedPoint);
	wq2.print_dist();
	std::cout << std::endl;
	WeightQuantizer wq3(2,2,4, DistType::FloatFixedPoint);
	wq3.print_dist();
	std::cout << std::endl;
	WeightQuantizer wq4(1,2,0.875, DistType::FixedPoint);
	wq4.print_dist();
	std::cout << std::endl;

	Quantizer q(1, 1, 2, 1);
	q.print_dist();
	std::cout << std::endl;
}

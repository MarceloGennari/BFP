#include "quantizer.h"

int main(){
	HardwareQuantizer hq(3);
	hq.print_dist();
	std::cout << std::endl;
	hq.set(4);
	hq.print_dist();
	std::cout << std::endl;
	hq.set(1);
	hq.print_dist();
	std::cout << std::endl;
	hq.set(8);
	hq.print_dist();
	std::cout << std::endl;

}

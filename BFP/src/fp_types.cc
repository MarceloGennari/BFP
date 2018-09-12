#include "fp_types.h"

FloatTypes::FloatTypes(int exponent_width, int mantissa_width, DistType dist) : e_w(exponent_width), m_w(mantissa_width), dist(dist) {};

IntTypes::IntTypes(int sizeBit, DistType dist) : sizeBit(sizeBit), dist(dist) {};

/*******************************
 NON-STATIC MEMBER FUNCTIONS
*******************************/
std::vector<float> FloatTypes::getDistribution(){
	return FloatTypes::getDistribution(dist, e_w, m_w);
}
std::vector<float> FloatTypes::getFixedPoint(){
	return FloatTypes::getFixedPoint(e_w+m_w+1); // The +1 is to account for negative numbers
}

std::vector<float> FloatTypes::getFloatingPoint(){
	return FloatTypes::getFloatingPoint(this->e_w, this->m_w);
}

std::vector<float> FloatTypes::getFloatFixedPoint(){
	return FloatTypes::getFloatFixedPoint(this->e_w, this->m_w);
}

std::vector<int> IntTypes::getDistribution(){
	return IntTypes::getDistribution(dist, this->sizeBit); 
}

std::vector<int> IntTypes::getInt(){
	return IntTypes::getInt(this->sizeBit);
}

/******************************
 STATIC MEMBER FUNCTIONS
******************************/
std::vector<float> FloatTypes::getDistribution(DistType d, int e_w, int m_w){
	switch(d){
		case DistType::FloatingPoint:   return getFloatingPoint(e_w, m_w);
		case DistType::FloatFixedPoint: return getFloatFixedPoint(e_w, m_w);
		case DistType::FixedPoint:	return getFixedPoint(e_w+m_w);
		default:			std::cout << "Not valid. Abort" << std::endl;
						exit(0);
	}
}

std::vector<float> FloatTypes::getDistribution(DistType d, int sizeBit){
	switch(d){
		case DistType::FloatingPoint: 	std::cout << "Just one argument provided for distribution" << std::endl;
						exit(0);
		case DistType::FloatFixedPoint: std::cout << "Just one argument provieded for distribution" << std::endl;
						exit(0);
		case DistType::FixedPoint: 	return getFixedPoint(sizeBit);
		default:			std::cout << "Not valid. Abort" << std::endl;
						exit(0);
	}
}

std::vector<int> IntTypes::getDistribution(DistType d, int sizeBit){
	switch(d){
		case DistType::INT: return getInt(sizeBit);
		default: std::cout << "Not valid. Abort" << std::endl;
			 exit(0);
	}
}

/********************
* PRIVATE MEMBER FUNCTIONS
*********************/
std::vector<int> IntTypes::getInt(int sizeBit){
	/*
	* This function returns a vector that has all the possible int values given the size bit
	*/
	std::vector<int> values;
	int numbBits = sizeBit;
	int maxV = std::pow(2, numbBits);
	
	for(int k = 1-maxV; k<maxV; k++){
		values.push_back(k);
	}
	return values;
}

std::vector<float> FloatTypes::getFixedPoint(int sizeBit){
	/*
	* This function returns a vector that has all the possible fixed point values given the e_w and m_w
	* The value is between [0, 1)
	*/
	std::vector<float> values;
	int numbBits = sizeBit;
	int maxV = std::pow(2, numbBits);
	
	for(int k = 1-maxV; k<maxV; k++){
		values.push_back(((float)k/maxV));	
		if(k==0){
			values.push_back((float)(k/maxV));
		}
	}
	return values;
}

std::vector<float> FloatTypes::getFloatingPoint(int e_w, int m_w){
	/*
	* This function returns a vector that has all the possible floating point values given the e_w, m_w
	* This follows the IEEE754 convention of (-1)^sign * 2^exp * (1.b0b1b2b3b4)
	*/
	std::vector<float> values;
	int minW = 0;
	int maxW = 1;

	if(e_w!=0){
		minW = -std::pow(2, e_w-1)+1;
		maxW = std::pow(2, e_w-1)+1;
	}	
	
	for(int w = minW; w < maxW; w++){
		for(int m = 0; m <std::pow(2, m_w); m++){
			float acc = 0;
			int n = m;
			std::string binary;
			while(n!=0) {binary=(n%2==0 ?"0":"1")+binary; n/=2;}
			for(int cnd = 0; cnd<binary.length(); cnd++){
				if(binary[cnd]=='1'){
					int value = -(binary.length() - cnd);
					acc+= std::pow(2, value);
					
				}
			}
		
			values.push_back(std::pow(2, w)*(1+acc));
			values.push_back((-1)*std::pow(2, w)*(1+acc));
		} 
	}
	values.push_back(0);
	std::sort(values.begin(), values.end());
	return values;
}

std::vector<float> FloatTypes::getFloatFixedPoint(int e_w, int m_w){
	/*
	* This function returns a vector that is floating point for all values apart from the lowest exponent
	* In the lowest exponent, this function assumes the value of a fixed point representation
	*/
		
	std::vector<float> values;
	int minW = 0;
	int maxW = 1;

	if(e_w!=0){
		minW = -std::pow(2, e_w-1)+1;
		maxW = std::pow(2, e_w-1)+1;
	}	
	
	for(int w = minW; w<maxW; w++){
		if(w==minW){
			std::vector<float> v = FloatTypes::getFixedPoint(m_w);
			for(auto it=v.begin(); it!=v.end(); it++){
				*it = *it*std::pow(2,minW+1);
			}
			values.insert(values.end(), v.begin(), v.end() );
			continue;			
		}
		for(int m = 0; m<std::pow(2,m_w); m++){
			float acc = 0;
			int n = m;
			std::string binary;
			while(n!=0) {binary=(n%2==0 ?"0":"1")+binary; n/=2;}
			for(int cnd=0; cnd<binary.length(); cnd++){
				if(binary[cnd] =='1'){
					int value = -(binary.length() - cnd);
					acc+= std::pow(2, value);
				}
			}
			
			values.push_back(std::pow(2, w)*(1+acc));
			values.push_back((-1)*std::pow(2, w)*(1+acc));
		}

	}
	std::sort(values.begin(), values.end());
	return values;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>
#include <algorithm>

enum class DistType {FixedPoint, FloatingPoint, FloatFixedPoint};

class FloatTypes{
	public:
		FloatTypes(int exponent_width, int mantissa_width, DistType = DistType::FixedPoint);
		
		
		static std::vector<float> getDistribution(DistType d, int e_w, int m_w);
		static std::vector<float> getDistribution(DistType d, int sizeBit);
		
		std::vector<float> getDistribution();
		std::vector<float> getFixedPoint();
		std::vector<float> getFloatFixedPoint();
		std::vector<float> getFloatingPoint();	

	private:
		int e_w;
		int m_w;
		DistType dist;
		std::vector<float> distribution;

		static std::vector<float> getFloatingPoint(int e_w, int m_w);
		static std::vector<float> getFixedPoint(int sizeBit);
		static std::vector<float> getFloatFixedPoint(int e_w, int m_w);
}; 

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <iostream>

using namespace boost::multiprecision;

// Set required precision here:
#define Precision 10

typedef number<cpp_dec_float<Precision>> large_float;

double CalculateLogProbability(const unsigned int M, const unsigned int s_M)
{
	// 0.5^M;
	large_float power = 0.5;
	power = pow(power, M);

	// sum (M choose k), k=s_M to M;
	large_float bin = boost::math::binomial_coefficient<large_float>(M, s_M);
	large_float sum = bin;
	for (large_float k = (s_M + 1); k <= M; k++)
	{
		// Use identity: (n choose k) = ((n-k+1)/k)(n choose (k-1))
		bin = bin * (((large_float)M - k + 1) / k);
		sum += bin;
	}

	// -log10(0.5^M * sum (M choose k), k=s_M to M)
	large_float result = -log10(power*sum);

	// convert to double precision
	return result.convert_to<double>();
}

int main(int argc, char* argv[])
{
	// How large is your validation set?
	const unsigned int M = 2000000;
	// How many traces in the validation set have been classified correctly?
	const unsigned int s_M = 1003017;

	std::cout << CalculateLogProbability(M, s_M) << std::endl;

	return 0;
}
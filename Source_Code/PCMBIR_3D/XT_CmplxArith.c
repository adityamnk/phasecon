
#include <math.h>
#include "XT_Constants.h"

void cmplx_mult (Real_t* real, Real_t* imag, Real_t x, Real_t y, Real_t a, Real_t b)
{
	*real = x*a - y*b;
	*imag = x*b + y*a;
}

void cmplx_div (Real_t* real, Real_t* imag, Real_t x, Real_t y, Real_t a, Real_t b)
{
	Real_t mag;

	cmplx_mult (real, imag, x, y, a, -b);
	mag = a*a + b*b;
		
	*real /= (mag + EPSILON_ERROR);
	*imag /= (mag + EPSILON_ERROR);
}


#include <stdio.h>
#include <fftw3.h>
#include <math.h>
#include "XT_Structures.h"
#include "XT_CmplxArith.h"
#include "XT_Constants.h"

/*Compute y = Hx where H is the Fresnel transform*/
void compute_FresnelTran (int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan)
{
	int32_t i, j;

#ifndef FAR_FIELD_DIFFRACTION
	Real_t u, v, real, imag, kover2R, phase, dist, multfact; 
	int32_t N;
	dist = FRESNEL_DISTANCE_FRACTION*(delta_rows*delta_cols)/LIGHT_WAVELENGTH;
	kover2R = M_PI/(dist*LIGHT_WAVELENGTH);
	multfact = 1.0/(2.0*kover2R*delta_rows*delta_cols);
	N = rows*cols;

	fftw_execute(*fftforw_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		u = i*(2*M_PI)/(rows*delta_rows);
		v = j*(2*M_PI)/(cols*delta_cols);
		phase = -(u*u + v*v)/(4*kover2R) + M_PI/2;
		cmplx_mult(&real, &imag, fftforw_arr[i*cols + j][0], fftforw_arr[i*cols + j][1], cos(phase), sin(phase));
		fftback_arr[i*cols + j][0] = real*multfact/N;
		fftback_arr[i*cols + j][1] = imag*multfact/N;
	}
	
	fftw_execute(*fftback_plan);
#else
	fftw_execute(*fftforw_plan);
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		fftback_arr[i*cols + j][0] = fftforw_arr[i*cols + j][0];
		fftback_arr[i*cols + j][1] = fftforw_arr[i*cols + j][1];
	}
#endif
}


/*Compute y = Hx where H is the Fresnel transform*/
void compute_HermFresnelTran (int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan)
{
	int32_t i, j;

#ifndef FAR_FIELD_DIFFRACTION
	Real_t u, v, real, imag, kover2R, phase, dist, multfact; int32_t N;
	dist = FRESNEL_DISTANCE_FRACTION*(delta_rows*delta_cols)/LIGHT_WAVELENGTH;
	kover2R = M_PI/(dist*LIGHT_WAVELENGTH);
	multfact = 1.0/(2.0*kover2R*delta_rows*delta_cols);
	N = rows*cols;

	fftw_execute(*fftforw_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		u = i*(2*M_PI)/(rows*delta_rows);
		v = j*(2*M_PI)/(cols*delta_cols);
		phase = (u*u + v*v)/(4*kover2R) - M_PI/2;
		cmplx_mult(&real, &imag, fftforw_arr[i*cols + j][0], fftforw_arr[i*cols + j][1], cos(phase), sin(phase));
		fftback_arr[i*cols + j][0] = real*multfact/N;
		fftback_arr[i*cols + j][1] = imag*multfact/N;
	}
	
	fftw_execute(*fftback_plan);

#else
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		fftback_arr[i*cols + j][0] = fftforw_arr[i*cols + j][0];
		fftback_arr[i*cols + j][1] = fftforw_arr[i*cols + j][1];
	}
	fftw_execute(*fftback_plan);

#endif
}

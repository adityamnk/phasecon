#include <stdio.h>
#include <fftw3.h>
#include <math.h>
#include "XT_Structures.h"
#include "XT_CmplxArith.h"
#include "XT_Constants.h"
#include "allocate.h"
#include "XT_IOMisc.h"
#include "XT_MPIIO.h"

/*Compute y = Hx where H is the Fresnel transform*/
void compute_FresnelTran (int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan, Real_t light_wavelength, Real_t obj2det_distance)
{
	int32_t i, j;

#ifndef FAR_FIELD_DIFFRACTION
	Real_t u, v, real, imag, kover2R, phase, cosphase, sinphase;
/*	Real_arr_t **img; 
	int32_t N, dimTiff[4];*/
/*	dist = FRESNEL_DISTANCE_FRACTION*(delta_rows*delta_cols)/LIGHT_WAVELENGTH;*/
	kover2R = M_PI/(obj2det_distance*light_wavelength);
	Real_t N = rows*cols;

	fftw_execute(*fftforw_plan);
	
	/*img = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, rows, cols);*/

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		if (i >= rows/2)
			u = -(rows - i)*(2*M_PI)/(rows*delta_rows);
		else
			u = i*(2*M_PI)/(rows*delta_rows);
		
		if (j >= cols/2)
			v = -(cols - j)*(2*M_PI)/(cols*delta_cols);
		else
			v = j*(2*M_PI)/(cols*delta_cols);
		phase = -(u*u + v*v)/(4*kover2R);
		cosphase = cos(phase);
		sinphase = sin(phase);
	
/*		img[i][j] = sqrt(cosphase*cosphase + sinphase*sinphase);*/
			
		cmplx_mult(&real, &imag, fftforw_arr[i*cols + j][0], fftforw_arr[i*cols + j][1], cosphase, sinphase);
		fftback_arr[i*cols + j][0] = real/N;
		fftback_arr[i*cols + j][1] = imag/N;
	}
		
/*	dimTiff[0] = 1; dimTiff[1] = 1; dimTiff[2] = rows; dimTiff[3] = cols;
	WriteMultiDimArray2Tiff ("fresnel_freq", dimTiff, 0, 1, 2, 3, &(img[0][0]), 0, 0, 1, stdout);
    	write_SharedBinFile_At ("fresnel_freq", &(img[0][0]), 0, rows*cols, stdout);
	multifree(img, 2);*/
	
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
void compute_HermFresnelTran (int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan, Real_t light_wavelength, Real_t obj2det_distance)
{
	int32_t i, j;

#ifndef FAR_FIELD_DIFFRACTION
	Real_t u, v, real, imag, kover2R, phase, cosphase, sinphase;
/*	Real_arr_t **img; 
	int32_t N, dimTiff[4];*/
	/*dist = FRESNEL_DISTANCE_FRACTION*(delta_rows*delta_cols)/LIGHT_WAVELENGTH;*/
	kover2R = M_PI/(obj2det_distance*light_wavelength);
	Real_t N = rows*cols;

	fftw_execute(*fftforw_plan);
	
/*	img = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, rows, cols);*/

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		if (i >= rows/2)
			u = -(rows - i)*(2*M_PI)/(rows*delta_rows);
		else
			u = i*(2*M_PI)/(rows*delta_rows);
		
		if (j >= cols/2)
			v = -(cols - j)*(2*M_PI)/(cols*delta_cols);
		else
			v = j*(2*M_PI)/(cols*delta_cols);
		phase = (u*u + v*v)/(4*kover2R);
		cosphase = cos(phase);
		sinphase = sin(phase);
	
/*		img[i][j] = sqrt(cosphase*cosphase + sinphase*sinphase);*/
			
		cmplx_mult(&real, &imag, fftforw_arr[i*cols + j][0], fftforw_arr[i*cols + j][1], cosphase, sinphase);
		fftback_arr[i*cols + j][0] = real/N;
		fftback_arr[i*cols + j][1] = imag/N;
	}
		
/*	dimTiff[0] = 1; dimTiff[1] = 1; dimTiff[2] = rows; dimTiff[3] = cols;
	WriteMultiDimArray2Tiff ("fresnel_freq", dimTiff, 0, 1, 2, 3, &(img[0][0]), 0, 0, 1, stdout);
    	write_SharedBinFile_At ("fresnel_freq", &(img[0][0]), 0, rows*cols, stdout);
	multifree(img, 2);*/
	
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

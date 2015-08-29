#include <stdio.h>
#include <fftw3.h>
#include "XT_Constants.h"
#include "XT_Structures.h"
#include <math.h>

void paganins_phase_retrieval (Real_arr_t** measurements, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** projlength, Real_arr_t** z_real, Real_arr_t** z_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan)
{
	Real_t I0, tran, dist, u, v, thick;
	int32_t i, j;

	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			I0 = D_real[i][j]*D_real[i][j] + D_imag[i][j]*D_imag[i][j];
			fftforw_arr[i*cols + j][0] = measurements[i][j]/(I0*exp(-ATT_COEF_1*projlength[i][j]));
			fftforw_arr[i*cols + j][0] = fftforw_arr[i*cols + j][0]*fftforw_arr[i*cols + j][0]; 
			fftforw_arr[i*cols + j][1] = 0; 
		}		
	
	fftw_execute(*fftforw_plan);

	dist = FRESNEL_DISTANCE_FRACTION*(delta_rows*delta_cols)/LIGHT_WAVELENGTH;
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		u = i*(2*M_PI)/(rows*delta_rows);
		v = j*(2*M_PI)/(cols*delta_cols);
		tran = dist*((REF_IND_DEC_2 - REF_IND_DEC_1)/(ATT_COEF_2 - ATT_COEF_1))*(u*u + v*v) + 1;
		tran = 1.0/(tran*delta_rows*delta_cols);
		fftback_arr[i*cols + j][0] = fftforw_arr[i*cols + j][0]*tran;
		fftback_arr[i*cols + j][1] = fftforw_arr[i*cols + j][1]*tran;
	}
	
	fftw_execute(*fftback_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		thick = -log(fftback_arr[i*cols + j][0])/(ATT_COEF_2 - ATT_COEF_1);
		z_real[i][j] = ATT_COEF_2*thick + ATT_COEF_1*(projlength[i][j] - thick);
		z_imag[i][j] = LIGHT_WAVENUMBER*(REF_IND_DEC_2*thick + REF_IND_DEC_1*(projlength[i][j] - thick));
	}	
}

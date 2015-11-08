#include <stdio.h>
#include <fftw3.h>
#include "XT_Constants.h"
#include "XT_Structures.h"
#include <math.h>

void paganins_phase_retrieval (Real_arr_t** measurements, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t*  projlength, Real_arr_t** z_real, Real_arr_t** z_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan, Real_t light_wavenumber, Real_t light_wavelength, Real_t obj2det_dist, Real_t pag_regparam)
{
	Real_t I0, tran, u, v, thick;
	int32_t i, j, N;

	/*alpha = (REF_IND_DEC_2 + REF_IND_DEC_1)/(ATT_COEF_2 + ATT_COEF_1);*/
	
	N = rows*cols;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			I0 = D_real[i][j]*D_real[i][j] + D_imag[i][j]*D_imag[i][j];
/*			fftforw_arr[i*cols + j][0] = measurements[i][j]*measurements[i][j]/(I0*exp(-ATT_COEF_1*projlength[i]));*/
			fftforw_arr[i*cols + j][0] = measurements[i][j]*measurements[i][j]/(I0);
			fftforw_arr[i*cols + j][1] = 0; 
		}		
	
	fftw_execute(*fftforw_plan);

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		if (i >= rows/2)
			u = -(rows - i)/(rows*delta_rows);
		else
			u = i/(rows*delta_rows);
		
		if (j >= cols/2)
			v = -(cols - j)/(cols*delta_cols);
		else
			v = j/(cols*delta_cols);
	/*	u = i/(rows*delta_rows);
		v = j/(cols*delta_cols);*/
		tran = obj2det_dist*(pag_regparam)*(u*u + v*v) + 1;
		tran = 1.0/(tran*N);
		fftback_arr[i*cols + j][0] = fftforw_arr[i*cols + j][0]*tran;
		fftback_arr[i*cols + j][1] = fftforw_arr[i*cols + j][1]*tran;
	}
	
	fftw_execute(*fftback_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		/*thick = -log(fabs(fftback_arr[i*cols + j][0]))/(ATT_COEF_2 - ATT_COEF_1);*/
		thick = -log(fabs(fftback_arr[i*cols + j][0]));
		/*printf("i = %d, j = %d, thickness = %f, proj length = %f\n", i, j, thick, projlength[i]);*/
/*		z_real[i][j] = LIGHT_WAVENUMBER*(ABSORP_COEF_2*thick + ABSORP_COEF_1*(projlength[i] - thick));
		z_imag[i][j] = LIGHT_WAVENUMBER*(REF_IND_DEC_2*thick + REF_IND_DEC_1*(projlength[i] - thick));*/
		z_real[i][j] = light_wavenumber*thick*light_wavelength/(4*M_PI);
/*		z_real[i][j] = light_wavenumber*(ATT_COEF_2 + ATT_COEF_1)*(projlength[i] - thick/(ATT_COEF_2 + ATT_COEF_1))/(2*light_wavenumber);*/
		z_imag[i][j] = light_wavenumber*thick*pag_regparam;
	}	
}

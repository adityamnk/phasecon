#include <stdio.h>
#include <fftw3.h>
#include "XT_Constants.h"
#include "XT_Structures.h"
#include <math.h>

void paganins_2mat_phase_retrieval (Real_arr_t** measurements, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t**  projlength, Real_arr_t** z_real, Real_arr_t** z_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan, Real_t light_wavenumber, Real_t light_wavelength, Real_t obj2det_dist, Real_t pag_regparam)
{
	Real_t I0, tran, u, v, thick, att_1, att_2;
	int32_t i, j, N;
	
	att_2 = ABSORP_COEF_2*4*M_PI/light_wavelength;
	att_1 = ABSORP_COEF_1*4*M_PI/light_wavelength;

	pag_regparam = (REF_IND_DEC_2 - REF_IND_DEC_1)/(att_2 - att_1); 
	printf("pag_regparam = %f, obj2det_dist*pag_regparam = %f\n", pag_regparam, pag_regparam*obj2det_dist);

	N = rows*cols;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			I0 = D_real[i][j]*D_real[i][j] + D_imag[i][j]*D_imag[i][j];
			fftforw_arr[i*cols + j][0] = measurements[i][j]*measurements[i][j]/(I0*exp(-att_1*projlength[i][j]));
			/*fftforw_arr[i*cols + j][0] = measurements[i][j]*measurements[i][j]/(I0);*/
			fftforw_arr[i*cols + j][1] = 0; 
#ifdef EXTRA_DEBUG_MESSAGES
		printf("i = %d, j = %d, fftforw real = %e, fftforw imag = %e\n", i, j, fftforw_arr[i*cols + j][0], fftforw_arr[i*cols + j][1]);
#endif
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
/*		tran = obj2det_dist*(pag_regparam)*(u*u + v*v)*4*M_PI*M_PI + 1;*/
		tran = obj2det_dist*(pag_regparam)*(u*u + v*v) + 1;
		tran = 1.0/(tran*N);
		fftback_arr[i*cols + j][0] = fftforw_arr[i*cols + j][0]*tran;
		fftback_arr[i*cols + j][1] = fftforw_arr[i*cols + j][1]*tran;
#ifdef EXTRA_DEBUG_MESSAGES
		printf("i = %d, j = %d, tran = %e, tran*N = %e, fftback_arr real = %e, fftback_arr imag = %e\n", i, j, tran, tran*N, fftback_arr[i*cols + j][0], fftback_arr[i*cols + j][1]);
#endif
	}
	
	fftw_execute(*fftback_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		thick = -log(fabs(fftback_arr[i*cols + j][0]))/(att_2 - att_1);
/*		if (thick < 0) thick = 0;*/
/*		thick = -log(fabs(fftback_arr[i*cols + j][0]));*/
#ifdef EXTRA_DEBUG_MESSAGES
		printf("i = %d, j = %d, thickness = %e, proj length = %e, fftback real = %e, fftback imag = %e\n", i, j, thick, projlength[i][j], fftback_arr[i*cols + j][0], fftback_arr[i*cols + j][1]);
#endif
		z_real[i][j] = light_wavenumber*(ABSORP_COEF_2*thick + ABSORP_COEF_1*(projlength[i][j] - thick));
		z_imag[i][j] = light_wavenumber*(REF_IND_DEC_2*thick + REF_IND_DEC_1*(projlength[i][j] - thick));
		/*z_real[i][j] = light_wavenumber*thick*light_wavelength/(4*M_PI);*/
/*		z_real[i][j] = 0;*/
/*		z_real[i][j] = light_wavenumber*(ATT_COEF_2 + ATT_COEF_1)*(projlength[i] - thick/(ATT_COEF_2 + ATT_COEF_1))/(2*light_wavenumber);*/
/*		z_imag[i][j] = light_wavenumber*thick*pag_regparam;*/
	}
}	

void paganins_1mat_phase_retrieval (Real_arr_t** measurements, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** z_real, Real_arr_t** z_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan, Real_t light_wavenumber, Real_t light_wavelength, Real_t obj2det_dist, Real_t delta_over_beta)
{
	Real_t I0, tran, u, v, thick, pag_regparam;
	int32_t i, j, N;
	
	pag_regparam = delta_over_beta/(2*light_wavenumber);
	N = rows*cols;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			I0 = D_real[i][j]*D_real[i][j] + D_imag[i][j]*D_imag[i][j];
			fftforw_arr[i*cols + j][0] = measurements[i][j]*measurements[i][j]/(I0);
			/*fftforw_arr[i*cols + j][0] = measurements[i][j]*measurements[i][j]/(I0);*/
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
		tran = obj2det_dist*(pag_regparam)*(u*u + v*v)*4*M_PI*M_PI + 1;
/*		tran = obj2det_dist*(pag_regparam)*(u*u + v*v) + 1;*/
		tran = 1.0/(tran*N);
		fftback_arr[i*cols + j][0] = fftforw_arr[i*cols + j][0]*tran;
		fftback_arr[i*cols + j][1] = fftforw_arr[i*cols + j][1]*tran;
	}
	
	fftw_execute(*fftback_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		thick = -log(fabs(fftback_arr[i*cols + j][0]));
	/*	if (thick < 0) thick = 0;*/
/*		thick = -log(fabs(fftback_arr[i*cols + j][0]));*/
#ifdef EXTRA_DEBUG_MESSAGES
		/*printf("i = %d, j = %d, thickness = %f, proj length = %f, fftback real = %f, fftback imag = %f\n", i, j, thick, projlength[i][j], fftback_arr[i*cols + j][0], fftback_arr[i*cols + j][1]);*/
#endif
/*		z_real[i][j] = light_wavenumber*(ABSORP_COEF_2*thick + ABSORP_COEF_1*(projlength[i][j] - thick));
		z_imag[i][j] = light_wavenumber*(REF_IND_DEC_2*thick + REF_IND_DEC_1*(projlength[i][j] - thick));*/
		z_real[i][j] = thick/2;
/*		z_real[i][j] = 0;*/
/*		z_real[i][j] = light_wavenumber*(ATT_COEF_2 + ATT_COEF_1)*(projlength[i] - thick/(ATT_COEF_2 + ATT_COEF_1))/(2*light_wavenumber);*/
		z_imag[i][j] = thick*delta_over_beta/2;
	}
}	

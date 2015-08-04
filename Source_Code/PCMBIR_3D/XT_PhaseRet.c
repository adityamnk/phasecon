#include <stdio.h>
#include "XT_Constants.h"
#include "allocate.h"
#include "XT_Structures.h"
#include <math.h>
#include <fftw3.h>
#include "XT_CmplxArith.h"

void compute_phase_projection (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, int32_t rows, int32_t cols, fftw_complex* fftarr, fftw_plan* p) 
{
	int32_t i, j;
	Real_t real, imag, mag;
	
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			cmplx_mult (&(real), &(imag), D_real[i][j], D_imag[i][j], w_real[i][j], w_imag[i][j]);
			fftarr[i*cols + j][0] = real; 
			fftarr[i*cols + j][1] = imag; 
		}

	fftw_execute(*p);

	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			cmplx_div (&(real), &(imag), y_real[i][j], y_imag[i][j], fftarr[i*cols + j][0], fftarr[i*cols + j][1]);
			mag = sqrt(pow(real, 2) + pow(imag, 2));
			Omega_real[i][j] = real/(mag + EPSILON_ERROR);	
			Omega_imag[i][j] = imag/(mag + EPSILON_ERROR);	
		}
}

void mult_Herm_OmegaHD (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, int32_t rows, int32_t cols, fftw_complex* fftarr, fftw_plan* p)
{
	int32_t i, j;
	Real_t real, imag;

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		cmplx_mult (&(real), &(imag), Omega_real[i][j], -Omega_imag[i][j], y_real[i][j], y_imag[i][j]);
		fftarr[i*cols + j][0] = real;
		fftarr[i*cols + j][1] = imag;
	}

	fftw_execute(*p);

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
		cmplx_mult (&(y_real[i][j]), &(y_imag[i][j]), fftarr[i*cols + j][0], fftarr[i*cols + j][1], D_real[i][j], -D_imag[i][j]);
}		


void mult_OmegaHD (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, int32_t rows, int32_t cols, fftw_complex *fftarr, fftw_plan *p)
{
	int32_t i, j;
	Real_t real, imag;

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		cmplx_mult (&(real), &(imag), D_real[i][j], D_imag[i][j], y_real[i][j], y_imag[i][j]);
		fftarr[i*cols + j][0] = real;
		fftarr[i*cols + j][1] = imag;
	}

	fftw_execute(*p);

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
		cmplx_mult (&(y_real[i][j]), &(y_imag[i][j]), fftarr[i*cols + j][0], fftarr[i*cols + j][1], Omega_real[i][j], Omega_imag[i][j]);
}		

void compute_gradient (Real_arr_t** g_real, Real_arr_t** g_imag, Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** Lambda, Real_arr_t** v_real, Real_arr_t** v_imag, Real_t nu, int32_t rows, int32_t cols, fftw_complex *fftforward_arr, fftw_plan *fftforward_plan, fftw_complex *fftbackward_arr, fftw_plan *fftbackward_plan)
{
	int32_t i, j;
	Real_arr_t **buf1_real, **buf1_imag, **buf2_real, **buf2_imag;

	buf1_real = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols);	
	buf1_imag = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols);	
	buf2_real = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols);	
	buf2_imag = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols);	

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		buf1_real[i][j] = w_real[i][j];
		buf1_imag[i][j] = w_imag[i][j];
	}
	
	mult_OmegaHD (buf1_real, buf1_imag, Omega_real, Omega_imag, D_real, D_imag, rows, cols, fftforward_arr, fftforward_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		buf1_real[i][j] *= Lambda[i][j];
		buf1_imag[i][j] *= Lambda[i][j];
	}		

	mult_Herm_OmegaHD (buf1_real, buf1_imag, Omega_real, Omega_imag, D_real, D_imag, rows, cols, fftbackward_arr, fftbackward_plan);

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		buf2_real[i][j] = Lambda[i][j]*y_real[i][j];
		buf2_imag[i][j] = Lambda[i][j]*y_imag[i][j];
	}		

	mult_Herm_OmegaHD (buf2_real, buf2_imag, Omega_real, Omega_imag, D_real, D_imag, rows, cols, fftbackward_arr, fftbackward_plan);
		
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		g_real[i][j] = buf1_real[i][j] - buf2_real[i][j] + nu*(w_real[i][j] - v_real[i][j]);
		g_imag[i][j] = buf1_imag[i][j] - buf2_imag[i][j] + nu*(w_imag[i][j] - v_imag[i][j]);
	}

	multifree(buf1_real, 2);
	multifree(buf1_imag, 2);
	multifree(buf2_real, 2);
	multifree(buf2_imag, 2);
}

Real_t compute_stepsize (Real_arr_t** g_real, Real_arr_t** g_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** Lambda, Real_t nu, int32_t rows, int32_t cols, fftw_complex *fftforward_arr, fftw_plan *fftforward_plan, fftw_complex *fftbackward_arr, fftw_plan *fftbackward_plan)
{
	int32_t i, j;
	Real_t alpha, acc_num = 0, acc_den = 0, **buf_real, **buf_imag;
	
	buf_real = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols);
	buf_imag = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols);

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		acc_num += g_real[i][j]*g_real[i][j] + g_imag[i][j]*g_imag[i][j];	
		buf_real[i][j] = g_real[i][j];
		buf_imag[i][j] = g_imag[i][j];
	}

	mult_OmegaHD (buf_real, buf_imag, Omega_real, Omega_imag, D_real, D_imag, rows, cols, fftforward_arr, fftforward_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
                buf_real[i][j] *= Lambda[i][j];
                buf_imag[i][j] *= Lambda[i][j];	
	}

	mult_Herm_OmegaHD (buf_real, buf_imag, Omega_real, Omega_imag, D_real, D_imag, rows, cols, fftbackward_arr, fftbackward_plan);
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
		acc_den += (g_real[i][j]*buf_real[i][j] + g_imag[i][j]*buf_imag[i][j]);

	acc_den += nu*acc_num;

	/*Adding EPSILON_ERROR to get rid of divide by 0 issues. */
	alpha = acc_num/(acc_den + EPSILON_ERROR);

	multifree(buf_real, 2);
	multifree(buf_imag, 2);
	return (alpha);
}

Real_t compute_cost (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** Lambda, Real_arr_t** v_real, Real_arr_t** v_imag, Real_t nu, int32_t rows, int32_t cols, fftw_complex *fftarr, fftw_plan *p)
{
	int32_t i, j;
	Real_t cost1 = 0, cost2 = 0, real, imag;
	Real_arr_t **buf_real, **buf_imag;

	buf_real = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols); 
	buf_imag = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols); 

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		buf_real[i][j] = w_real[i][j];
		buf_imag[i][j] = w_imag[i][j];
	}
	
	mult_OmegaHD (buf_real, buf_imag, Omega_real, Omega_imag, D_real, D_imag, rows, cols, fftarr, p);	

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		cmplx_mult (&(real), &(imag), y_real[i][j] - buf_real[i][j], y_imag[i][j] - buf_imag[i][j], y_real[i][j] - buf_real[i][j], -(y_imag[i][j] - buf_imag[i][j]));
		cost1 += real*Lambda[i][j];
	}
	
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		cmplx_mult (&(real), &(imag), w_real[i][j]-v_real[i][j], w_imag[i][j]-v_imag[i][j],  w_real[i][j]-v_real[i][j], -(w_imag[i][j]-v_imag[i][j]));
		cost2 += real;
	}	

	multifree(buf_real,2);	
	multifree(buf_imag,2);	

	return (cost1/2 + nu*cost2/2);
}

Real_t steepest_descent_iter (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** Lambda, Real_arr_t** v_real, Real_arr_t** v_imag, Real_t nu, int32_t rows, int32_t cols, fftw_complex *fftforward_arr, fftw_plan *fftforward_plan, fftw_complex *fftbackward_arr, fftw_plan *fftbackward_plan)
{
	int32_t i, j;
	Real_arr_t **g_real, **g_imag, alpha, gavg_real = 0, gavg_imag = 0, wavg_real = 0, wavg_imag = 0;
	Real_t cost_old, cost_new, upavg = 0, valavg = 0;

	g_real = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols); 
	g_imag = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, rows, cols); 

	cost_old = compute_cost (y_real, y_imag, Omega_real, Omega_imag, D_real, D_imag, w_real, w_imag, Lambda, v_real, v_imag, nu, rows, cols, fftforward_arr, fftforward_plan);
	compute_gradient (g_real, g_imag, y_real, y_imag, Omega_real, Omega_imag, D_real, D_imag, w_real, w_imag, Lambda, v_real, v_imag, nu, rows, cols, fftforward_arr, fftforward_plan, fftbackward_arr, fftbackward_plan);
	alpha = compute_stepsize (g_real, g_imag, Omega_real, Omega_imag, D_real, D_imag, Lambda, nu, rows, cols, fftforward_arr, fftforward_plan, fftbackward_arr, fftbackward_plan);

/*	printf("Stepsize alpha = %f\n", alpha);*/

	for (i = 0; i < rows; i++)	
	for (j = 0; j < cols; j++)
	{
		w_real[i][j] += -alpha*g_real[i][j];
		w_imag[i][j] += -alpha*g_imag[i][j];
		upavg += sqrt(alpha*alpha*(g_real[i][j]*g_real[i][j] + g_imag[i][j]*g_imag[i][j]));
		valavg += sqrt(w_real[i][j]*w_real[i][j] + w_imag[i][j]*w_imag[i][j]);
	}	
	upavg /= valavg;
		
	for (i = 0; i < rows; i++)	
	for (j = 0; j < cols; j++)
	{
		wavg_real += fabs(w_real[i][j]);
		wavg_imag += fabs(w_imag[i][j]);
		gavg_real += fabs(g_real[i][j]);
		gavg_imag += fabs(g_imag[i][j]);
	}
/*	printf("Average magnitude of gradient: real = %f, imag = %f\n", gavg_real, gavg_imag);
	printf("Average magnitude of estimate of w: real = %f, imag = %f\n", wavg_real, wavg_imag);*/

	cost_new = compute_cost (y_real, y_imag, Omega_real, Omega_imag, D_real, D_imag, w_real, w_imag, Lambda, v_real, v_imag, nu, rows, cols, fftforward_arr, fftforward_plan);

	if (cost_new > cost_old)
	{
/*		printf("cost_old = %f, cost_new = %f\n", cost_old, cost_new);
		printf("ERROR: Cost increased after w update!\n");*/
	}
	cost_old = cost_new;

/*	compute_phase_projection (y, Omega_real, Omega_imag, D_real, D_imag, w_real, w_imag, rows, cols, fftforward_arr, fftforward_plan); 
	cost_new = compute_cost (y, Omega_real, Omega_imag, D_real, D_imag, w_real, w_imag, Lambda, v_real, v_imag, nu, rows, cols, fftforward_arr, fftforward_plan);
	if (cost_new > cost_old)
	{
		printf("cost_old = %f, cost_new = %f\n", cost_old, cost_new);
		printf("ERROR: Cost increased after Omega update!\n");
	}*/
	
	multifree(g_real, 2);
	multifree(g_imag, 2);

	return(upavg*100);	
} 

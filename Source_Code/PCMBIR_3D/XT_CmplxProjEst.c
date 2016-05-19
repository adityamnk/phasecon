#include <stdio.h>
#include <fftw3.h>
#include "XT_Structures.h"
#include "allocate.h"
#include "XT_Constants.h"
#include <math.h>
#include "XT_Search.h"
#include "XT_PhaseRet.h"
#include "XT_CmplxArith.h"
#include "XT_IOMisc.h"
#include "XT_Prior.h"
#include "XT_FresnelTran.h"
#include "XT_MPIIO.h"

Real_t compute_pretcost (Real_arr_t** measurements_real, Real_arr_t** measurements_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t*** z_real, Real_arr_t*** z_imag, Real_arr_t** Lambda, Real_arr_t** proj_real, Real_arr_t** proj_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, Real_t mu, fftw_complex *fftforw_arr, fftw_plan *fftforw_plan, fftw_complex *fftback_arr, fftw_plan *fftback_plan, Real_t light_wavelength, Real_t obj2det_distance, Real_arr_t** FreqWindow, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	int32_t j, k;
	Real_t exptemp, real, imag, cost = 0, forw_cost = 0, prior_cost = 0, diff_delta, diff_beta, diff;	

	for (j = 0; j < rows; j++)	
	for (k = 0; k < cols; k++)
	{
		exptemp = exp(-z_real[j][k][1]);
		cmplx_mult(&real, &imag, exptemp*cos(-z_imag[j][k][1]), exptemp*sin(-z_imag[j][k][1]), D_real[j][k], D_imag[j][k]);
		fftforw_arr[j*cols + k][0] = real;
		fftforw_arr[j*cols + k][1] = imag;
	}
	
	compute_FresnelTran (rows, cols, delta_rows, delta_cols, fftforw_arr, fftforw_plan, fftback_arr, fftback_plan, light_wavelength, obj2det_distance, FreqWindow);

	for (j = 0; j < rows; j++)	
	for (k = 0; k < cols; k++)
	{
		real = (measurements_real[j][k] - sqrt(pow(fftback_arr[j*cols + k][0],2) + pow(fftback_arr[j*cols + k][1],2)));
		forw_cost += real*real*Lambda[j][k]/2;
	}		

#ifdef ENABLE_TOMO_RECONS
	for (j = 0; j < rows; j++)	
	for (k = 0; k < cols; k++)
		prior_cost += mu*((z_real[j][k][1] - proj_real[j][k])*(z_real[j][k][1] - proj_real[j][k]) + (z_imag[j][k][1] - proj_imag[j][k])*(z_imag[j][k][1] - proj_imag[j][k]))/2;
#else
	for (j = 0; j < rows; j++)
		for (k = 0; k < cols; k++)
		{
			if (k + 1 < cols)
			{
				diff_beta = z_real[j][k][1] - z_real[j][k+1][1];
				diff_delta = z_imag[j][k][1] - z_imag[j][k+1][1];
				diff = ScannedObjectPtr->DecorrTran[0][0]*diff_delta + ScannedObjectPtr->DecorrTran[0][1]*diff_beta;
          			prior_cost += TomoInputsPtr->Spatial_Filter[1][1][2]*Phase_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);
				diff = ScannedObjectPtr->DecorrTran[1][0]*diff_delta + ScannedObjectPtr->DecorrTran[1][1]*diff_beta;
          			prior_cost += TomoInputsPtr->Spatial_Filter[1][1][2]*Mag_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);
			}
			if (j + 1 < rows)
			{
				diff_beta = z_real[j][k][1] - z_real[j+1][k][1];
				diff_delta = z_imag[j][k][1] - z_imag[j+1][k][1];
				diff = ScannedObjectPtr->DecorrTran[0][0]*diff_delta + ScannedObjectPtr->DecorrTran[0][1]*diff_beta;
          			prior_cost += TomoInputsPtr->Spatial_Filter[1][2][1]*Phase_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);
				diff = ScannedObjectPtr->DecorrTran[1][0]*diff_delta + ScannedObjectPtr->DecorrTran[1][1]*diff_beta;
          			prior_cost += TomoInputsPtr->Spatial_Filter[1][2][1]*Mag_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);	
				if (k + 1 < cols)
				{
					diff_beta = z_real[j][k][1] - z_real[j+1][k+1][1];
					diff_delta = z_imag[j][k][1] - z_imag[j+1][k+1][1];
					diff = ScannedObjectPtr->DecorrTran[0][0]*diff_delta + ScannedObjectPtr->DecorrTran[0][1]*diff_beta;
          				prior_cost += TomoInputsPtr->Spatial_Filter[1][2][2]*Phase_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);
					diff = ScannedObjectPtr->DecorrTran[1][0]*diff_delta + ScannedObjectPtr->DecorrTran[1][1]*diff_beta;
          				prior_cost += TomoInputsPtr->Spatial_Filter[1][2][2]*Mag_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);
				}
				if (k - 1 >= 0)
				{
					diff_beta = z_real[j][k][1] - z_real[j+1][k-1][1];
					diff_delta = z_imag[j][k][1] - z_imag[j+1][k-1][1];
					diff = ScannedObjectPtr->DecorrTran[0][0]*diff_delta + ScannedObjectPtr->DecorrTran[0][1]*diff_beta;
          				prior_cost += TomoInputsPtr->Spatial_Filter[1][2][0]*Phase_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);
					diff = ScannedObjectPtr->DecorrTran[1][0]*diff_delta + ScannedObjectPtr->DecorrTran[1][1]*diff_beta;
          				prior_cost += TomoInputsPtr->Spatial_Filter[1][2][0]*Mag_QGGMRF_Spatial_Value(diff, ScannedObjectPtr, TomoInputsPtr);
				}
			}
		}
#endif

	cost = forw_cost + prior_cost;
	printf("Forward cost = %e, prior cost = %e, cost = %e\n", forw_cost, prior_cost, cost);

	return (cost);	
}

void estimate_complex_projection (Real_arr_t** measurements_real, Real_arr_t** measurements_imag, Real_arr_t** omega_real, Real_arr_t** omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t*** z_real, Real_arr_t*** z_imag, Real_arr_t** Lambda, Real_arr_t** proj_real, Real_arr_t** proj_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** v_real, Real_arr_t** v_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, Real_t NMS_rho, Real_t NMS_chi, Real_t NMS_gamma, Real_t NMS_sigma, Real_t NMS_thresh, int32_t NMS_iters, Real_t steepdes_thresh, int32_t steepdes_iters, Real_t pret_thresh, int32_t pret_iters, Real_t mu, Real_t nu, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan, Real_t light_wavelength, Real_t obj2det_distance, Real_arr_t** FresnelFreqWin, int32_t proj_idx, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	char primalres_filename[100], dualres_filename[100], pretcost_filename[100];
	int32_t j, k, iter, NMS_avgiter, m, n, idxm, idxn, size, offset; 
	Real_arr_t **buf_real, **buf_imag, b_real, b_imag, **wold_real, **wold_imag;
	Real_t thresh = 0, cost, cost_old, cost_init, cost_NMS_old, cost_NMS, sum = 0, real_temp, imag_temp, primal_res, dual_res, zr_neigh[3][3], zi_neigh[3][3];
	bool bd_flag[3][3];

	sprintf(primalres_filename, "%s_proj_%d", PHASERET_PRIMAL_RESIDUAL_FILENAME, proj_idx);
	sprintf(dualres_filename, "%s_proj_%d", PHASERET_DUAL_RESIDUAL_FILENAME, proj_idx);
	sprintf(pretcost_filename, "%s_proj_%d", PHASERET_COST_FILENAME, proj_idx);

	buf_real = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, rows, cols);
	buf_imag = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, rows, cols);
	wold_real = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, rows, cols);
	wold_imag = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, rows, cols);

  	memset(&(v_real[0][0]), 0, rows*cols*sizeof(Real_arr_t));
  	memset(&(v_imag[0][0]), 0, rows*cols*sizeof(Real_arr_t));

	cost_init = compute_pretcost (measurements_real, measurements_imag, omega_real, omega_imag, D_real, D_imag, z_real, z_imag, Lambda, proj_real, proj_imag, rows, cols, delta_rows, delta_cols, mu, fftforw_arr, fftforw_plan, fftback_arr, fftback_plan, light_wavelength, obj2det_distance, FresnelFreqWin, ScannedObjectPtr, TomoInputsPtr);
	Write2Bin (pretcost_filename, 1, 1, 1, 1, sizeof(Real_t), &cost_init, TomoInputsPtr->debug_file_ptr);
	cost_old = cost_init;

	for (iter = 0; iter < pret_iters; iter++)
	{
		NMS_avgiter = 0;
		for (j = 0; j < rows; j++)
		for (k = 0; k < cols; k++)
		{
			b_real = w_real[j][k] - v_real[j][k]; 
			b_imag = w_imag[j][k] - v_imag[j][k];
			
#ifndef ENABLE_TOMO_RECONS
			for (m = 0; m < 3; m++)
			{
				idxm = j + m - 1;		
				if (idxm >= 0 && idxm < rows)
				{
					for (n = 0; n < 3; n++)
					{
						idxn = k + n - 1;
						if (idxn >= 0 && idxn < cols)
						{
							zr_neigh[m][n] = z_real[idxm][idxn][1];
							zi_neigh[m][n] = z_imag[idxm][idxn][1];
							bd_flag[m][n] = true;
						}
						else
						{
							zr_neigh[m][n] = 0;
							zi_neigh[m][n] = 0;
							bd_flag[m][n] = false;
						}
					}
				}
				else
				{
					for (n = 0; n < 3; n++)
					{
						zr_neigh[m][n] = 0;
						zi_neigh[m][n] = 0;
						bd_flag[m][n] = false;
					}
				} 
			}
#endif
		
			cost_NMS_old = Cost_NMS (z_real[j][k][1], z_imag[j][k][1], proj_real[j][k], proj_imag[j][k], b_real, b_imag, mu, nu, zr_neigh, zi_neigh, bd_flag, ScannedObjectPtr, TomoInputsPtr);

			NMS_avgiter += Nelder_Mead_Simplex_2DSearch (z_real[j][k], z_imag[j][k], proj_real[j][k], proj_imag[j][k], b_real, b_imag, NMS_rho, NMS_chi, NMS_gamma, NMS_sigma, NMS_thresh, NMS_iters, mu, nu, stdout, zr_neigh, zi_neigh, bd_flag, ScannedObjectPtr, TomoInputsPtr);
		
			cost_NMS = Cost_NMS (z_real[j][k][1], z_imag[j][k][1], proj_real[j][k], proj_imag[j][k], b_real, b_imag, mu, nu, zr_neigh, zi_neigh, bd_flag, ScannedObjectPtr, TomoInputsPtr);
			if (cost_NMS > cost_NMS_old)
				printf("ERROR: NMS cost increased for %d row, %d column.\n", j, k);

		}
		printf("Average number of NMS iterations is %f\n", (float)NMS_avgiter/(rows*cols));
		
		for (j = 0; j < rows; j++)
		for (k = 0; k < cols; k++)
		{
			wold_real[j][k] = w_real[j][k];
			wold_imag[j][k] = w_imag[j][k];
			buf_real[j][k] = exp(-z_real[j][k][1])*cos(-z_imag[j][k][1]) + v_real[j][k];	
			buf_imag[j][k] = exp(-z_real[j][k][1])*sin(-z_imag[j][k][1]) + v_imag[j][k];	
		}

		for (j = 0; j < steepdes_iters; j++)
		{
			thresh = steepest_descent_iter (measurements_real, measurements_imag, omega_real, omega_imag, D_real, D_imag, w_real, w_imag, Lambda, buf_real, buf_imag, nu, rows, cols, delta_rows, delta_cols, fftforw_arr, fftforw_plan, fftback_arr, fftback_plan, light_wavelength, obj2det_distance, FresnelFreqWin);
			if (thresh < steepdes_thresh && j > 1) break;
		}
		printf("Number of steepest descent iterations is %f, thresh = %f\n", (float)j, thresh);

		primal_res = 0; dual_res = 0; sum = 0;	
		for (j = 0; j < rows; j++)
		for (k = 0; k < cols; k++)
		{
			real_temp = exp(-z_real[j][k][1])*cos(-z_imag[j][k][1]) - w_real[j][k];
			imag_temp = exp(-z_real[j][k][1])*sin(-z_imag[j][k][1]) - w_imag[j][k];
			v_real[j][k] += real_temp;
			v_imag[j][k] += imag_temp;
			primal_res += real_temp*real_temp + imag_temp*imag_temp;
			dual_res += (w_real[j][k] - wold_real[j][k])*(w_real[j][k] - wold_real[j][k]) + (w_imag[j][k] - wold_imag[j][k])*(w_imag[j][k] - wold_imag[j][k]);
			sum += v_real[j][k]*v_real[j][k] + v_imag[j][k]*v_imag[j][k];
		}
		dual_res = sqrt(dual_res/sum);
		primal_res = sqrt(primal_res/(rows*cols));
	
	    	Append2Bin (primalres_filename, 1, 1, 1, 1, sizeof(Real_t), &primal_res, stdout);
	    	Append2Bin (dualres_filename, 1, 1, 1, 1, sizeof(Real_t), &dual_res, stdout);
		
		cost = compute_pretcost (measurements_real, measurements_imag, omega_real, omega_imag, D_real, D_imag, z_real, z_imag, Lambda, proj_real, proj_imag, rows, cols, delta_rows, delta_cols, mu, fftforw_arr, fftforw_plan, fftback_arr, fftback_plan, light_wavelength, obj2det_distance, FresnelFreqWin, ScannedObjectPtr, TomoInputsPtr);
	   	Append2Bin (pretcost_filename, 1, 1, 1, 1, sizeof(Real_t), &cost, TomoInputsPtr->debug_file_ptr);
		
		if (cost > cost_old)
			printf("WARNING: Cost increased! Old cost = %e, New cost = %e.\n", cost_old, cost);

		printf("----- PRet iters = %d, Proj idx = %d, Primal & dual residual = %e & %e. Old cost = %e, new cost = %e. Decrease in cost = %e\n ----", iter, proj_idx, primal_res, dual_res, cost_old, cost, cost_old-cost);
		cost_old = cost;
	
		if (dual_res < pret_thresh && iter > 1)
			break;
	
/*		size = rows*cols*4; offset = proj_idx*rows*cols*4; 
		write_SharedBinFile_At (MAGTOMOAUX_FILENAME, &(z_real[0][0][0]), offset, size, TomoInputsPtr->debug_file_ptr); 
		write_SharedBinFile_At (PHASETOMOAUX_FILENAME, &(z_imag[0][0][0]), offset, size, TomoInputsPtr->debug_file_ptr);*/
	}
	
	printf("**** PRet total iters is %d. Proj idx = %d. Primal and dual residual are %e and %e. Init cost is %e and new cost is %e. Total reduction in cost is %e ****\n", iter, proj_idx, primal_res, dual_res, cost_init, cost, cost_init-cost);
	if (cost > cost_init)
		printf("WARNING: Cost increased after phase retrieval! Old cost = %e, New cost = %e.\n", cost_init, cost);

	multifree(buf_real, 2);
	multifree(buf_imag, 2);
	multifree(wold_real, 2);
	multifree(wold_imag, 2);
}

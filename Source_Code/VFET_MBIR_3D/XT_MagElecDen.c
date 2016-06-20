/* ============================================================================
 * Copyright (c) 2013 K. Aditya Mohan (Purdue University)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * Neither the name of K. Aditya Mohan, Purdue
 * University, nor the names of its contributors may be used
 * to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#include <stdio.h>
#include "XT_Structures.h"
#include "XT_Prior.h"
#include "XT_Debug.h"
#include "XT_DensityUpdate.h"
#include "allocate.h"
#include <math.h>

Real_t computeADMMDensityCost(ScannedObject* ObjPtr, TomoInputs* InpPtr);

void reconstruct_magelec (ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr)
{
	int32_t i, Iter, j, k;		
	Real_t MagUpdate = 0, ElecUpdate = 0, MagSum = 0, ElecSum = 0, alpha_mag, alpha_elec, cost, cost_old;
	Real_arr_t ****grad_mag, ***grad_elec;

	grad_mag = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
	grad_elec = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);

	cost_old = computeADMMDensityCost(ObjPtr, InpPtr);
	fprintf(InpPtr->debug_file_ptr, "------------ Steep Grad Descent Iter = 0, cost = %e --------------\n", cost_old);
	for (Iter = 1; Iter < InpPtr->DensUpdate_MaxIter; Iter++)
	{
		compute_gradient_stepsize(ObjPtr, InpPtr, fftptr, grad_mag, grad_elec, &alpha_mag, &alpha_elec);
		
		MagUpdate = 0; ElecUpdate = 0; MagSum = 0; ElecSum = 0;
		for (i = 0; i < ObjPtr->N_z; i++)
		for (j = 0; j < ObjPtr->N_y; j++)
		for (k = 0; k < ObjPtr->N_x; k++)
		{
			ObjPtr->Magnetization[i][j][k][0] -= alpha_mag*grad_mag[i][j][k][0];			
			ObjPtr->Magnetization[i][j][k][1] -= alpha_mag*grad_mag[i][j][k][1];			
			ObjPtr->Magnetization[i][j][k][2] -= alpha_mag*grad_mag[i][j][k][2];			
			ObjPtr->ChargeDensity[i][j][k] -= alpha_elec*grad_elec[i][j][k];			

			MagUpdate += sqrt(pow(alpha_mag*grad_mag[i][j][k][0],2) + pow(alpha_mag*grad_mag[i][j][k][1],2) + pow(alpha_mag*grad_mag[i][j][k][2],2)); 
			ElecUpdate += fabs(alpha_elec*grad_elec[i][j][k]);
 
			MagSum += sqrt(pow(ObjPtr->Magnetization[i][j][k][0],2) + pow(ObjPtr->Magnetization[i][j][k][1],2) + pow(ObjPtr->Magnetization[i][j][k][2],2)); 
			ElecSum += fabs(ObjPtr->ChargeDensity[i][j][k]);
		}

		MagUpdate = MagUpdate*100/(MagSum + EPSILON_ERROR);
		ElecUpdate = ElecUpdate*100/(ElecSum + EPSILON_ERROR);

  		compute_crossprodtran (ObjPtr->Magnetization, ObjPtr->ChargeDensity, ObjPtr->ErrorPotMag, ObjPtr->ErrorPotElec, ObjPtr->MagFilt, ObjPtr->ElecFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 1);
		for (i = 0; i < ObjPtr->N_z; i++)
		for (j = 0; j < ObjPtr->N_y; j++)
		for (k = 0; k < ObjPtr->N_x; k++)
		{
			ObjPtr->ErrorPotMag[i][j][k][0] = ObjPtr->MagPotentials[i][j][k][0] - ObjPtr->MagPotDual[i][j][k][0] - ObjPtr->ErrorPotMag[i][j][k][0];
			ObjPtr->ErrorPotMag[i][j][k][1] = ObjPtr->MagPotentials[i][j][k][1] - ObjPtr->MagPotDual[i][j][k][1] - ObjPtr->ErrorPotMag[i][j][k][1];
			ObjPtr->ErrorPotMag[i][j][k][2] = ObjPtr->MagPotentials[i][j][k][2] - ObjPtr->MagPotDual[i][j][k][2] - ObjPtr->ErrorPotMag[i][j][k][2];
			ObjPtr->ErrorPotElec[i][j][k] = ObjPtr->ElecPotentials[i][j][k] - ObjPtr->ElecPotDual[i][j][k] - ObjPtr->ErrorPotElec[i][j][k];
		}

		cost = computeADMMDensityCost(ObjPtr, InpPtr);
		if (cost > cost_old)
	      		check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ERROR: Cost increased when updating magnetization and charge density.\n");
		cost_old = cost;
		
		fprintf(InpPtr->debug_file_ptr, "------------ Steep Grad Descent Iter = %d, cost = %e, Avg update as percentage (mag, elec) = (%e, %e), update = (%e,%e), sum  = (%e,%e)--------------\n", Iter, cost, MagUpdate, ElecUpdate, MagUpdate, ElecUpdate, MagSum, ElecSum);
		if (Iter > 1 && MagUpdate < InpPtr->DensUpdate_thresh && ElecUpdate < InpPtr->DensUpdate_thresh)
		{
			fprintf(InpPtr->debug_file_ptr, "******* Steepest gradient descent algorithm has converged! *********\n");
			break;		
		}
	}

	multifree(grad_mag, 4);
	multifree(grad_elec, 3);
}





Real_t computeADMMDensityCost(ScannedObject* ObjPtr, TomoInputs* InpPtr)
{
  Real_t cost=0, forward=0, prior=0;
  Real_t Diff;
  int32_t j,k,p,cidx,slice;
  bool j_minus, k_minus, j_plus, k_plus, p_plus; 
  
/*  #pragma omp parallel for private(j, k, sino_idx, slice)*/
    for (slice=0; slice<ObjPtr->N_z; slice++)
    {
    	for (j=0; j<ObjPtr->N_y; j++)
    	{
     		for (k=0; k<ObjPtr->N_x; k++)
		{
			forward += InpPtr->ADMM_mu*ObjPtr->ErrorPotMag[slice][j][k][0]*ObjPtr->ErrorPotMag[slice][j][k][0];
			forward += InpPtr->ADMM_mu*ObjPtr->ErrorPotMag[slice][j][k][1]*ObjPtr->ErrorPotMag[slice][j][k][1];
			forward += InpPtr->ADMM_mu*ObjPtr->ErrorPotMag[slice][j][k][2]*ObjPtr->ErrorPotMag[slice][j][k][2];
			forward += InpPtr->ADMM_mu*ObjPtr->ErrorPotElec[slice][j][k]*ObjPtr->ErrorPotElec[slice][j][k];
          	}
        }
    }
  
  forward /= 2.0;
 

  /*When computing the cost of the prior term it is important to make sure that you don't include the cost of any pair of neighbors more than once. In this code, a certain sense of causality is used to compute the cost. We also assume that the weghting kernel given by 'Filter' is symmetric. Let i, j and k correspond to the three dimensions. If we go forward to i+1, then all neighbors at j-1, j, j+1, k+1, k, k-1 are to be considered. However, if for the same i, if we go forward to j+1, then all k-1, k, and k+1 should be considered. For same i and j, only the neighbor at k+1 is considred.*/
  prior = 0;
/*  #pragma omp parallel for private(Diff, p, j, k, j_minus, k_minus, p_plus, j_plus, k_plus, cidx) reduction(+:prior)*/
  for (p = 0; p < ObjPtr->N_z; p++)
  for (j = 0; j < ObjPtr->N_y; j++)
  {
    for (k = 0; k < ObjPtr->N_x; k++)
    {
      j_minus = (j - 1 >= 0)? true : false;
      k_minus = (k - 1 >= 0)? true : false;
      
      p_plus = (p + 1 < ObjPtr->N_z)? true : false;
      j_plus = (j + 1 < ObjPtr->N_y)? true : false;
      k_plus = (k + 1 < ObjPtr->N_x)? true : false;
      
    if(k_plus == true) {
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = (ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p][j][k + 1][cidx]);
        	prior += InpPtr->Spatial_Filter[1][1][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	}
       Diff = (ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p][j][k + 1]);
        prior += InpPtr->Spatial_Filter[1][1][2] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
      }
      if(j_plus == true) {
        if(k_minus == true) {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = (ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p][j + 1][k - 1][cidx]);
          	prior += InpPtr->Spatial_Filter[1][2][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
          }
	  Diff = (ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p][j + 1][k - 1]);
          prior += InpPtr->Spatial_Filter[1][2][0] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
        }
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = (ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p][j + 1][k][cidx]);
        	prior += InpPtr->Spatial_Filter[1][2][1] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	}
        Diff = (ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p][j + 1][k]);
        prior += InpPtr->Spatial_Filter[1][2][1] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);

        if(k_plus == true) {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = (ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p][j + 1][k + 1][cidx]);
          	prior += InpPtr->Spatial_Filter[1][2][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
          }
	  Diff = (ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p][j + 1][k + 1]);
          prior += InpPtr->Spatial_Filter[1][2][2] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
        }
      }
      if (p_plus == true)
      {
        if(j_minus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j - 1][k][cidx];
          	prior += InpPtr->Spatial_Filter[2][0][1] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	  }
          Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p + 1][j - 1][k];
          prior += InpPtr->Spatial_Filter[2][0][1] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
        }
        
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p+1][j][k][cidx];
        	prior += InpPtr->Spatial_Filter[2][1][1] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	}
        Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p+1][j][k];
        prior += InpPtr->Spatial_Filter[2][1][1] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
        if(j_plus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p+1][j + 1][k][cidx];
          	prior += InpPtr->Spatial_Filter[2][2][1] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	  }
          Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p+1][j + 1][k];
          prior += InpPtr->Spatial_Filter[2][2][1] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
        }
        if(j_minus == true)
        {
          if(k_minus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j - 1][k - 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][0][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
            Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p + 1][j - 1][k - 1];
            prior += InpPtr->Spatial_Filter[2][0][0] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
          }
          if(k_plus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j - 1][k + 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][0][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
            Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p + 1][j - 1][k + 1];
            prior += InpPtr->Spatial_Filter[2][0][2] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
          }
        }
        if(k_minus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j][k - 1][cidx];
          	prior += InpPtr->Spatial_Filter[2][1][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	  }
          Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p + 1][j][k - 1];
          prior += InpPtr->Spatial_Filter[2][1][0] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
        }
        if(j_plus == true)
        {
          if(k_minus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j + 1][k - 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][2][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
            Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p + 1][j + 1][k - 1];
            prior += InpPtr->Spatial_Filter[2][2][0] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
          }
          if(k_plus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j + 1][k + 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][2][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
            Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p + 1][j + 1][k + 1];
            prior += InpPtr->Spatial_Filter[2][2][2] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
          }
        }
        if(k_plus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j][k + 1][cidx];
          	prior += InpPtr->Spatial_Filter[2][1][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	  }
          Diff = ObjPtr->ChargeDensity[p][j][k] - ObjPtr->ChargeDensity[p + 1][j][k + 1];
          prior += InpPtr->Spatial_Filter[2][1][2] * QGGMRF_Value(Diff,InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C);
        }
      }
    }
    } 
    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Density Update Forward cost = %f\n",forward);
    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Density Update Prior cost = %f\n",prior);
    cost = forward + prior;
  
  return cost;
}


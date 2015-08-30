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

#include "XT_Constants.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "allocate.h"
#include "randlib.h"
#include <time.h>
#include "XT_AMatrix.h"
#include "XT_Profile.h"
#include "XT_Structures.h"
#include "XT_IOMisc.h"
#include "XT_NHICD.h"
#include "omp.h"
#include "XT_MPI.h"
#include <mpi.h>
#include "XT_VoxUpdate.h"
#include "XT_ForwardProject.h"
#include "XT_MPIIO.h"
#include "XT_Debug.h"
#include "XT_OffsetError.h"
#include "XT_Prior.h"
#include "XT_Search.h"
#include "XT_PhaseRet.h"
#include "XT_CmplxArith.h"
#include "XT_CmplxProjEst.h"
#include "XT_PhaseRet.h"
#include "XT_FresnelTran.h"
/*computes the location of (i,j,k) th element in a 1D array*/
int32_t array_loc_1D (int32_t i, int32_t j, int32_t k, int32_t N_j, int32_t N_k)
{
  return (i*N_j*N_k + j*N_k + k);
}

/*converts the value 'val' to hounsfield units and returns it*/
Real_t convert2Hounsfield (Real_t val)
{
  Real_t slope, c;
  
  slope=(HOUNSFIELD_WATER_MAP-HOUNSFIELD_AIR_MAP)/(WATER_MASS_ATT_COEFF*WATER_DENSITY-AIR_MASS_ATT_COEFF*AIR_DENSITY)/HFIELD_UNIT_CONV_CONST;
  c=-slope*(AIR_MASS_ATT_COEFF*AIR_DENSITY*HFIELD_UNIT_CONV_CONST);
  
  return (slope*val + c);
}

/*computes the value of cost function. 'ErrorSino' is the error sinogram*/
Real_t computeCost(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  Real_t cost=0,temp=0, forward=0, prior=0;
  Real_t delta;
  int32_t i,j,k,p,N_z;
  bool j_minus, k_minus, i_plus, j_plus, k_plus, p_plus;
  
  #pragma omp parallel for private(j, k, temp) reduction(+:cost)
  for (i = 0; i < SinogramPtr->N_p; i++)
  for (j = 0; j < SinogramPtr->N_r; j++)
  for (k = 0; k < SinogramPtr->N_t; k++)
  {
    temp = SinogramPtr->MagErrorSino[i][j][k]*TomoInputsPtr->ADMM_mu;
    cost += temp*temp;
    temp = SinogramPtr->PhaseErrorSino[i][j][k]*TomoInputsPtr->ADMM_mu;
    cost += temp*temp;
  }
  cost /= 2.0;
  /*When computing the cost of the prior term it is important to make sure that you don't include the cost of any pair of neighbors more than once. In this code, a certain sense of causality is used to compute the cost. We also assume that the weghting kernel given by 'Filter' is symmetric. Let i, j and k correspond to the three dimensions. If we go forward to i+1, then all neighbors at j-1, j, j+1, k+1, k, k-1 are to be considered. However, if for the same i, if we go forward to j+1, then all k-1, k, and k+1 should be considered. For same i and j, only the neighbor at k+1 is considred.*/
  temp = 0;
  N_z = ScannedObjectPtr->N_z + 2;
  if (TomoInputsPtr->node_rank == TomoInputsPtr->node_num-1)
  N_z = ScannedObjectPtr->N_z + 1;
  #pragma omp parallel for private(delta, p, j, k, j_minus, k_minus, p_plus, i_plus, j_plus, k_plus) reduction(+:temp)
  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  for (p = 1; p < ScannedObjectPtr->N_z + 1; p++)
  for (j = 0; j < ScannedObjectPtr->N_y; j++)
  {
    for (k = 0; k < ScannedObjectPtr->N_x; k++)
    {
      j_minus = (j - 1 >= 0)? true : false;
      k_minus = (k - 1 >= 0)? true : false;
      
      p_plus = (p + 1 < N_z)? true : false;
      i_plus = (i + 1 < ScannedObjectPtr->N_time)? true : false;
      j_plus = (j + 1 < ScannedObjectPtr->N_y)? true : false;
      k_plus = (k + 1 < ScannedObjectPtr->N_x)? true : false;
      
      if(k_plus == true) {
        delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j][k + 1]);
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j][k + 1]);
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
      }
      if(j_plus == true) {
        if(k_minus == true) {
          delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k - 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
          delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k - 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        }
        delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k]);
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k]);
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        if(k_plus == true) {
          delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k + 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
          delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k + 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        }
      }
      if (p_plus == true)
      {
        if(j_minus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
        
        delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j][k];
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j][k];
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        if(j_plus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j + 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j + 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_minus == true)
        {
          if(k_minus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_minus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k - 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k - 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_plus == true)
        {
          if(k_minus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_plus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k + 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k + 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
      }
      if(i_plus == true) {
        delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i+1][p][j][k]);
        temp += TomoInputsPtr->Time_Filter[0] * Mag_QGGMRF_Temporal_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i+1][p][j][k]);
        temp += TomoInputsPtr->Time_Filter[0] * Phase_QGGMRF_Temporal_Value(delta,ScannedObjectPtr,TomoInputsPtr);
      }
    }
  }
  /*Use MPI reduction operation to add the forward and prior costs from all nodes*/
  MPI_Reduce(&cost, &forward, 1, MPI_REAL_DATATYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&temp, &prior, 1, MPI_REAL_DATATYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (TomoInputsPtr->node_rank == 0)
  {
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Scaled error sino cost = %f\n",forward);
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Decrease in scaled error sino cost = %f\n",TomoInputsPtr->ErrorSino_Cost-forward);
    TomoInputsPtr->ErrorSino_Cost = forward;
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Forward cost = %f\n",forward);
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Prior cost = %f\n",prior);
    TomoInputsPtr->Forward_Cost = forward;
    TomoInputsPtr->Prior_Cost = prior;
    cost = forward + prior;
  }
  
  /*Broadcase the value of cost to all nodes*/
  MPI_Bcast(&cost, 1, MPI_REAL_DATATYPE, 0, MPI_COMM_WORLD);
  return cost;
}


/*computes the value of cost function. 'ErrorSino' is the error sinogram*/
Real_t compute_original_cost(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  Real_t cost=0,temp=0, forward=0, prior=0;
  Real_t delta, magtemp, costemp, sintemp; 
  Real_t ***real, ***imag;
  int32_t i,j,k,p,N_z,dimTiff[4];
  bool j_minus, k_minus, i_plus, j_plus, k_plus, p_plus;
  
  real = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
  imag = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
  #pragma omp parallel for private(j, k, magtemp, costemp, sintemp)
  for (i = 0; i < SinogramPtr->N_p; i++)
  {
	for (j = 0; j < SinogramPtr->N_r; j++)
  	for (k = 0; k < SinogramPtr->N_t; k++)
  	{
		magtemp = exp(SinogramPtr->MagErrorSino[i][j][k] - SinogramPtr->MagTomoAux[i][j][k][1] + SinogramPtr->MagTomoDual[i][j][k]);
    		costemp = magtemp*cos(SinogramPtr->PhaseErrorSino[i][j][k] - SinogramPtr->PhaseTomoAux[i][j][k][1] + SinogramPtr->PhaseTomoDual[i][j][k]);
    		sintemp = magtemp*sin(SinogramPtr->PhaseErrorSino[i][j][k] - SinogramPtr->PhaseTomoAux[i][j][k][1] + SinogramPtr->PhaseTomoDual[i][j][k]);
		cmplx_mult (&(real[i][j][k]), &(imag[i][j][k]), costemp, sintemp, SinogramPtr->D_real[i][j][k], SinogramPtr->D_imag[i][j][k]);
		SinogramPtr->fftforw_arr[i][j*SinogramPtr->N_t + k][0] = real[i][j][k];
		SinogramPtr->fftforw_arr[i][j*SinogramPtr->N_t + k][1] = imag[i][j][k];
    	}
  }

  if (TomoInputsPtr->Write2Tiff == 1)
  {
	dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
    	WriteMultiDimArray2Tiff ("cost_real_prefft", dimTiff, 0, 3, 1, 2, &(real[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
    	WriteMultiDimArray2Tiff ("cost_imag_prefft", dimTiff, 0, 3, 1, 2, &(imag[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  }

  #pragma omp parallel for private(j, k, magtemp, costemp, sintemp) reduction(+:cost)
  for (i = 0; i < SinogramPtr->N_p; i++)
  {
	compute_FresnelTran (SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]));
		
	for (j = 0; j < SinogramPtr->N_r; j++)
  	for (k = 0; k < SinogramPtr->N_t; k++)
  	{
		real[i][j][k] = SinogramPtr->Measurements_real[i][j][k] - sqrt(pow(SinogramPtr->fftback_arr[i][j*SinogramPtr->N_t + k][0], 2) + pow(SinogramPtr->fftback_arr[i][j*SinogramPtr->N_t + k][1], 2));
		imag[i][j][k] = 0;
	/*	cmplx_mult (&(costemp), &(sintemp), SinogramPtr->fftforw_arr[i][j*SinogramPtr->N_t + k][0], SinogramPtr->fftforw_arr[i][j*SinogramPtr->N_t + k][1], SinogramPtr->Omega_real[i][j][k], SinogramPtr->Omega_imag[i][j][k]);
		real[i][j][k] = SinogramPtr->Measurements_real[i][j][k] - costemp;
		imag[i][j][k] = SinogramPtr->Measurements_imag[i][j][k] - sintemp;*/
		cost += (real[i][j][k]*real[i][j][k] + imag[i][j][k]*imag[i][j][k])*TomoInputsPtr->Weight[i][j][k];
 	}
  }
  cost /= 2.0;
  
  if (TomoInputsPtr->Write2Tiff == 1)
  {
	dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
    	WriteMultiDimArray2Tiff ("cost_real_postfft", dimTiff, 0, 3, 1, 2, &(real[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
    	WriteMultiDimArray2Tiff ("cost_imag_postfft", dimTiff, 0, 3, 1, 2, &(imag[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  }
  /*When computing the cost of the prior term it is important to make sure that you don't include the cost of any pair of neighbors more than once. In this code, a certain sense of causality is used to compute the cost. We also assume that the weghting kernel given by 'Filter' is symmetric. Let i, j and k correspond to the three dimensions. If we go forward to i+1, then all neighbors at j-1, j, j+1, k+1, k, k-1 are to be considered. However, if for the same i, if we go forward to j+1, then all k-1, k, and k+1 should be considered. For same i and j, only the neighbor at k+1 is considred.*/
  temp = 0;
  N_z = ScannedObjectPtr->N_z + 2;
  if (TomoInputsPtr->node_rank == TomoInputsPtr->node_num-1)
  N_z = ScannedObjectPtr->N_z + 1;
  #pragma omp parallel for private(delta, p, j, k, j_minus, k_minus, p_plus, i_plus, j_plus, k_plus) reduction(+:temp)
  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  for (p = 1; p < ScannedObjectPtr->N_z + 1; p++)
  for (j = 0; j < ScannedObjectPtr->N_y; j++)
  {
    for (k = 0; k < ScannedObjectPtr->N_x; k++)
    {
      j_minus = (j - 1 >= 0)? true : false;
      k_minus = (k - 1 >= 0)? true : false;
      
      p_plus = (p + 1 < N_z)? true : false;
      i_plus = (i + 1 < ScannedObjectPtr->N_time)? true : false;
      j_plus = (j + 1 < ScannedObjectPtr->N_y)? true : false;
      k_plus = (k + 1 < ScannedObjectPtr->N_x)? true : false;
      
      if(k_plus == true) {
        delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j][k + 1]);
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j][k + 1]);
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
      }
      if(j_plus == true) {
        if(k_minus == true) {
          delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k - 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
          delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k - 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        }
        delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k]);
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k]);
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        if(k_plus == true) {
          delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k + 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Mag_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
          delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k + 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Phase_QGGMRF_Spatial_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        }
      }
      if (p_plus == true)
      {
        if(j_minus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
        
        delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j][k];
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j][k];
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        if(j_plus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j + 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j + 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_minus == true)
        {
          if(k_minus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_minus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k - 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k - 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_plus == true)
        {
          if(k_minus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
            delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_plus == true)
        {
          delta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k + 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Mag_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
          delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k + 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Phase_QGGMRF_Spatial_Value(delta, ScannedObjectPtr, TomoInputsPtr);
        }
      }
      if(i_plus == true) {
        delta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i+1][p][j][k]);
        temp += TomoInputsPtr->Time_Filter[0] * Mag_QGGMRF_Temporal_Value(delta,ScannedObjectPtr,TomoInputsPtr);
        delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i+1][p][j][k]);
        temp += TomoInputsPtr->Time_Filter[0] * Phase_QGGMRF_Temporal_Value(delta,ScannedObjectPtr,TomoInputsPtr);
      }
    }
  }
  /*Use MPI reduction operation to add the forward and prior costs from all nodes*/
  MPI_Reduce(&cost, &forward, 1, MPI_REAL_DATATYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&temp, &prior, 1, MPI_REAL_DATATYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (TomoInputsPtr->node_rank == 0)
  {
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Original Forward cost = %f\n",forward);
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Original Prior cost = %f\n",prior);
    cost = forward + prior;
  }
 
  multifree(real, 3); 
  multifree(imag, 3); 
  /*Broadcase the value of cost to all nodes*/
  MPI_Bcast(&cost, 1, MPI_REAL_DATATYPE, 0, MPI_COMM_WORLD);
  return cost;
}

/*Upsamples the (N_time x N_z x N_y x N_x) size 'Init' by a factor of 2 along the x-y plane and stores it in 'Object'*/
void upsample_bilinear_2D (Real_arr_t**** Object, Real_arr_t**** Init, int32_t N_time, int32_t N_z, int32_t N_y, int32_t N_x)
{
  int32_t i, j, k, m;
  Real_arr_t **buffer;
  
  #pragma omp parallel for private(buffer, m, j, k)
  for (i=0; i < N_time; i++)
  for (m=0; m < N_z; m++)
  {
    buffer = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, N_y, 2*N_x);
    for (j=0; j < N_y; j++){
      buffer[j][0] = Init[i][m][j][0];
      buffer[j][1] = (3.0*Init[i][m][j][0] + Init[i][m][j][1])/4.0;
      buffer[j][2*N_x - 1] = Init[i][m][j][N_x - 1];
      buffer[j][2*N_x - 2] = (Init[i][m][j][N_x - 2] + 3.0*Init[i][m][j][N_x - 1])/4.0;
      for (k=1; k < N_x - 1; k++){
        buffer[j][2*k] = (Init[i][m][j][k-1] + 3.0*Init[i][m][j][k])/4.0;
        buffer[j][2*k + 1] = (3.0*Init[i][m][j][k] + Init[i][m][j][k+1])/4.0;
      }
    }
    for (k=0; k < 2*N_x; k++){
      Object[i][m][0][k] = buffer[0][k];
      Object[i][m][1][k] = (3.0*buffer[0][k] + buffer[1][k])/4.0;
      Object[i][m][2*N_y-1][k] = buffer[N_y-1][k];
      Object[i][m][2*N_y-2][k] = (buffer[N_y-2][k] + 3.0*buffer[N_y-1][k])/4.0;
    }
    for (j=1; j<N_y-1; j++){
      for (k=0; k<2*N_x; k++){
        Object[i][m][2*j][k] = (buffer[j-1][k] + 3.0*buffer[j][k])/4.0;
        Object[i][m][2*j + 1][k] = (3*buffer[j][k] + buffer[j+1][k])/4.0;
      }
    }
    multifree(buffer,2);
  }
}
/*Upsamples the (N_z x N_y x N_x) size 'Init' by a factor of 2 along the x-y plane and stores it in 'Object'*/
void upsample_object_bilinear_2D (Real_arr_t*** Object, Real_arr_t*** Init, int32_t N_z, int32_t N_y, int32_t N_x)
{
  int32_t j, k, slice;
  Real_arr_t **buffer;
  
  
  buffer = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, N_y, 2*N_x);
  for (slice=0; slice < N_z; slice++){
    for (j=0; j < N_y; j++){
      buffer[j][0] = Init[slice][j][0];
      buffer[j][1] = (3.0*Init[slice][j][0] + Init[slice][j][1])/4.0;
      buffer[j][2*N_x - 1] = Init[slice][j][N_x - 1];
      buffer[j][2*N_x - 2] = (Init[slice][j][N_x - 2] + 3.0*Init[slice][j][N_x - 1])/4.0;
      for (k=1; k < N_x - 1; k++){
        buffer[j][2*k] = (Init[slice][j][k-1] + 3.0*Init[slice][j][k])/4.0;
        buffer[j][2*k + 1] = (3.0*Init[slice][j][k] + Init[slice][j][k+1])/4.0;
      }
    }
    for (k=0; k < 2*N_x; k++){
      Object[slice+1][0][k] = buffer[0][k];
      Object[slice+1][1][k] = (3.0*buffer[0][k] + buffer[1][k])/4.0;
      Object[slice+1][2*N_y-1][k] = buffer[N_y-1][k];
      Object[slice+1][2*N_y-2][k] = (buffer[N_y-2][k] + 3.0*buffer[N_y-1][k])/4.0;
    }
    for (j=1; j<N_y-1; j++){
      for (k=0; k<2*N_x; k++){
        Object[slice+1][2*j][k] = (buffer[j-1][k] + 3.0*buffer[j][k])/4.0;
        Object[slice+1][2*j + 1][k] = (3*buffer[j][k] + buffer[j+1][k])/4.0;
      }
    }
  }
  multifree(buffer,2);
}

void upsample_bilinear_3D (Real_arr_t**** Object, Real_arr_t**** Init, int32_t N_time, int32_t N_z, int32_t N_y, int32_t N_x)
{
  int32_t i, j, k, slice;
  Real_t ***buffer2D, ***buffer3D;
  
  #pragma omp parallel for private(buffer2D, buffer3D, slice, j, k)
  for (i=0; i < N_time; i++)
  {
	  buffer2D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, N_y, 2*N_x);
	  buffer3D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, 2*N_y, 2*N_x);
	  for (slice=0; slice < N_z; slice++){
	    for (j=0; j < N_y; j++){
	      buffer2D[slice][j][0] = Init[i][slice][j][0];
	      buffer2D[slice][j][1] = (3.0*Init[i][slice][j][0] + Init[i][slice][j][1])/4.0;
	      buffer2D[slice][j][2*N_x - 1] = Init[i][slice][j][N_x - 1];
	      buffer2D[slice][j][2*N_x - 2] = (Init[i][slice][j][N_x - 2] + 3.0*Init[i][slice][j][N_x - 1])/4.0;
	      for (k=1; k < N_x - 1; k++){
        	buffer2D[slice][j][2*k] = (Init[i][slice][j][k-1] + 3.0*Init[i][slice][j][k])/4.0;
        	buffer2D[slice][j][2*k + 1] = (3.0*Init[i][slice][j][k] + Init[i][slice][j][k+1])/4.0;
      	     }
    	    }
    	    for (k=0; k < 2*N_x; k++){
      		buffer3D[slice][0][k] = buffer2D[slice][0][k];
      		buffer3D[slice][1][k] = (3.0*buffer2D[slice][0][k] + buffer2D[slice][1][k])/4.0;
      		buffer3D[slice][2*N_y-1][k] = buffer2D[slice][N_y-1][k];
      		buffer3D[slice][2*N_y-2][k] = (buffer2D[slice][N_y-2][k] + 3.0*buffer2D[slice][N_y-1][k])/4.0;
    	    }
    		for (j=1; j<N_y-1; j++)
    		for (k=0; k<2*N_x; k++){
      			buffer3D[slice][2*j][k] = (buffer2D[slice][j-1][k] + 3.0*buffer2D[slice][j][k])/4.0;
      			buffer3D[slice][2*j + 1][k] = (3*buffer2D[slice][j][k] + buffer2D[slice][j+1][k])/4.0;
    		}
  	   }
  
  	for (j=0; j<2*N_y; j++)
  	for (k=0; k<2*N_x; k++){
    		Object[i][0][j][k] = buffer3D[0][j][k];
    		Object[i][1][j][k] = (3.0*buffer3D[0][j][k] + buffer3D[1][j][k])/4.0;
    		Object[i][2*N_z-1][j][k] = buffer3D[N_z-1][j][k];
    		Object[i][2*N_z-2][j][k] = (3.0*buffer3D[N_z-1][j][k] + buffer3D[N_z-2][j][k])/4.0;
  	}
  
  	for (slice=1; slice < N_z-1; slice++)
  	for (j=0; j<2*N_y; j++)
  	for (k=0; k<2*N_x; k++){
    		Object[i][2*slice][j][k] = (buffer3D[slice-1][j][k] + 3.0*buffer3D[slice][j][k])/4.0;
    		Object[i][2*slice+1][j][k] = (3.0*buffer3D[slice][j][k] + buffer3D[slice+1][j][k])/4.0;
  	}
  
  	multifree(buffer2D,3);
  	multifree(buffer3D,3);
  }
}

/*'InitObject' intializes the Object to be reconstructed to either 0 or an interpolated version of the previous reconstruction. It is used in multi resolution reconstruction in which after every coarse resolution reconstruction the object should be intialized with an interpolated version of the reconstruction following which the object will be reconstructed at a finer resolution.*/
/*Upsamples the (N_time x N_z x N_y x N_x) size 'Init' by a factor of 2 along the in 3D x-y-z coordinates and stores it in 'Object'*/
void upsample_object_bilinear_3D (Real_arr_t*** Object, Real_arr_t*** Init, int32_t N_z, int32_t N_y, int32_t N_x)
{
  int32_t j, k, slice;
  Real_t ***buffer2D, ***buffer3D;
  
  buffer2D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, N_y, 2*N_x);
  buffer3D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, 2*N_y, 2*N_x);
  for (slice=0; slice < N_z; slice++){
    for (j=0; j < N_y; j++){
      buffer2D[slice][j][0] = Init[slice][j][0];
      buffer2D[slice][j][1] = (3.0*Init[slice][j][0] + Init[slice][j][1])/4.0;
      buffer2D[slice][j][2*N_x - 1] = Init[slice][j][N_x - 1];
      buffer2D[slice][j][2*N_x - 2] = (Init[slice][j][N_x - 2] + 3.0*Init[slice][j][N_x - 1])/4.0;
      for (k=1; k < N_x - 1; k++){
        buffer2D[slice][j][2*k] = (Init[slice][j][k-1] + 3.0*Init[slice][j][k])/4.0;
        buffer2D[slice][j][2*k + 1] = (3.0*Init[slice][j][k] + Init[slice][j][k+1])/4.0;
      }
    }
    for (k=0; k < 2*N_x; k++){
      buffer3D[slice][0][k] = buffer2D[slice][0][k];
      buffer3D[slice][1][k] = (3.0*buffer2D[slice][0][k] + buffer2D[slice][1][k])/4.0;
      buffer3D[slice][2*N_y-1][k] = buffer2D[slice][N_y-1][k];
      buffer3D[slice][2*N_y-2][k] = (buffer2D[slice][N_y-2][k] + 3.0*buffer2D[slice][N_y-1][k])/4.0;
    }
    for (j=1; j<N_y-1; j++)
    for (k=0; k<2*N_x; k++){
      buffer3D[slice][2*j][k] = (buffer2D[slice][j-1][k] + 3.0*buffer2D[slice][j][k])/4.0;
      buffer3D[slice][2*j + 1][k] = (3*buffer2D[slice][j][k] + buffer2D[slice][j+1][k])/4.0;
    }
  }
  
  for (j=0; j<2*N_y; j++)
  for (k=0; k<2*N_x; k++){
    Object[1][j][k] = buffer3D[0][j][k];
    Object[2][j][k] = (3.0*buffer3D[0][j][k] + buffer3D[1][j][k])/4.0;
    Object[2*N_z][j][k] = buffer3D[N_z-1][j][k];
    Object[2*N_z-1][j][k] = (3.0*buffer3D[N_z-1][j][k] + buffer3D[N_z-2][j][k])/4.0;
  }
  
  for (slice=1; slice < N_z-1; slice++)
  for (j=0; j<2*N_y; j++)
  for (k=0; k<2*N_x; k++){
    Object[2*slice+1][j][k] = (buffer3D[slice-1][j][k] + 3.0*buffer3D[slice][j][k])/4.0;
    Object[2*slice+2][j][k] = (3.0*buffer3D[slice][j][k] + buffer3D[slice+1][j][k])/4.0;
  }
  
  multifree(buffer2D,3);
  multifree(buffer3D,3);
}

void dwnsmpl_object (Real_arr_t*** Object, float*** Init, int32_t N_z, int32_t N_y, int32_t N_x, int32_t dwnsmpl_z, int32_t dwnsmpl_y, int32_t dwnsmpl_x, int32_t interp)
{
	int32_t i, j, k, m, n, p;
	
	for (i = 0; i < N_z; i++)
	for (j = 0; j < N_y; j++)
	for (k = 0; k < N_x; k++)
	{
		Object[i][j][k] = 0;
		for (m = 0; m < dwnsmpl_z; m++)
		for (n = 0; n < dwnsmpl_y; n++)
		for (p = 0; p < dwnsmpl_x; p++)
		{
			if (interp == 0 && Object[i][j][k] > Init[i*dwnsmpl_z + m][j*dwnsmpl_y + n][k*dwnsmpl_x + p])/*downsample with minimum in neiborhood*/
				Object[i][j][k] = Init[i*dwnsmpl_z + m][j*dwnsmpl_y + n][k*dwnsmpl_x + p];
			else if (interp == 1 && Object[i][j][k] < Init[i*dwnsmpl_z + m][j*dwnsmpl_y + n][k*dwnsmpl_x + p])/*downsample with maximum in neiborhood*/
				Object[i][j][k] = Init[i*dwnsmpl_z + m][j*dwnsmpl_y + n][k*dwnsmpl_x + p];
			else if (interp == 2)
				Object[i][j][k] += Init[i*dwnsmpl_z + m][j*dwnsmpl_y + n][k*dwnsmpl_x + p];
		}
		
		if (interp == 2)
			Object[i][j][k] /= (dwnsmpl_z*dwnsmpl_y*dwnsmpl_x);
	}
}

int init_minmax_object (ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	float ***Init;
	FILE *fp;
	int32_t size, result;
	char maxobj_filename[] = MAX_OBJ_FILEPATH;
	char minobj_filename[] = MIN_OBJ_FILEPATH;
	int32_t dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, flag = 0;

      	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Initializing the min and max arrays (of object)...\n");
	size = PHANTOM_Z_SIZE*PHANTOM_XY_SIZE*PHANTOM_XY_SIZE/TomoInputsPtr->node_num;
	dwnsmpl_z = PHANTOM_Z_SIZE/(ScannedObjectPtr->N_z*TomoInputsPtr->node_num); 	
	dwnsmpl_y = PHANTOM_XY_SIZE/ScannedObjectPtr->N_y; 	
	dwnsmpl_x = PHANTOM_XY_SIZE/ScannedObjectPtr->N_x; 	

	Init = (float***)multialloc(sizeof(float), 3, PHANTOM_Z_SIZE/TomoInputsPtr->node_num, PHANTOM_XY_SIZE, PHANTOM_XY_SIZE);

	fp = fopen (minobj_filename, "rb");
	result = fseek (fp, TomoInputsPtr->node_rank*size*sizeof(float), SEEK_SET);
	result = fread (&(Init[0][0][0]), sizeof(float), size, fp);
	fclose (fp);
	dwnsmpl_object (ScannedObjectPtr->MagObjMin, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 0);
	dwnsmpl_object (ScannedObjectPtr->PhaseObjMin, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 0);

	fp = fopen (maxobj_filename, "rb");
	result = fseek (fp, TomoInputsPtr->node_rank*size*sizeof(float), SEEK_SET);
	result = fread (&(Init[0][0][0]), sizeof(float), size, fp);
	fclose (fp);
	dwnsmpl_object (ScannedObjectPtr->MagObjMax, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 1);
	dwnsmpl_object (ScannedObjectPtr->PhaseObjMax, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 1);

	multifree(Init, 3);
      	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Completed initialization of min and max arrays.\n");
	
	return (flag);	
}

/*randomly select the voxels lines which need to be updated along the x-y plane for each z-block and time slice*/
void randomly_select_x_y (ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, uint8_t*** Mask)
{
  int32_t i, j, num,n, Index, col, row, *Counter, ArraySize, block;
  ArraySize = ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
  Counter = (int32_t*)get_spc(ArraySize, sizeof(int32_t));
  for (i=0; i<ScannedObjectPtr->N_time; i++)
  for (block=0; block<TomoInputsPtr->num_z_blocks; block++)
  {
    ArraySize = ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
    for (Index = 0; Index < ArraySize; Index++)
    Counter[Index] = Index;
    
    TomoInputsPtr->UpdateSelectNum[i][block] = 0;
    for (j=0; j<ScannedObjectPtr->N_x*ScannedObjectPtr->N_y; j++){
      Index = floor(random2() * ArraySize);
      Index = (Index == ArraySize)?ArraySize-1:Index;
      col = Counter[Index] % ScannedObjectPtr->N_x;
      row = Counter[Index] / ScannedObjectPtr->N_x;
      for (n = block*(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks); n < (block+1)*(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks); n++)
      if (Mask[i][row][col] == 1)
      {
        num = TomoInputsPtr->UpdateSelectNum[i][block];
        TomoInputsPtr->x_rand_select[i][block][num] = col;
        TomoInputsPtr->y_rand_select[i][block][num] = row;
        (TomoInputsPtr->UpdateSelectNum[i][block])++;
        break;
      }
      Counter[Index] = Counter[ArraySize - 1];
      ArraySize--;
    }
  }
  free(Counter);
}

void init_GroundTruth (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	float ***Init;
	char object_file[100];
	int32_t dimTiff[4];
	Real_arr_t ***RealObj, ***ImagObj, ***RealSino, ***ImagSino;
	Real_t pixel; FILE *fp;
	int32_t N_z, N_y, N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, sino_idx, i, j, k, l, p, slice;
	long int stream_offset, size, result;
	char phantom_filename[] = PHANTOM_FILEPATH;
/*	char phantom_filename[] = MIN_OBJ_FILEPATH;*/ 
  	AMatrixCol* AMatrixPtr = (AMatrixCol*)get_spc(ScannedObjectPtr->N_time, sizeof(AMatrixCol));
  	uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
  
	for (i = 0; i < ScannedObjectPtr->N_time; i++)
	{
    		AMatrixPtr[i].values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
    		AMatrixPtr[i].index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  	}

	N_z = PHANTOM_Z_SIZE/TomoInputsPtr->node_num;
	N_y = PHANTOM_XY_SIZE;
	N_x = PHANTOM_XY_SIZE;
	size = N_z*N_y*N_x;	
	dwnsmpl_z = N_z / ScannedObjectPtr->N_z;
	dwnsmpl_y = N_y / ScannedObjectPtr->N_y;
	dwnsmpl_x = N_x / ScannedObjectPtr->N_x;
	Init = (float***)multialloc(sizeof(float), 3, N_z, N_y, N_x);
	RealObj = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	ImagObj = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	RealSino = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	ImagSino = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	
	fp = fopen (phantom_filename, "rb");
/*	check_error(fp==NULL, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Error in reading file %s\n", phantom_filename);*/
	result = fread (&(Init[0][0][0]), sizeof(float), size, fp);
/*  	check_error(result != size, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_filename,result,stream_offset,size);*/
	dwnsmpl_object (RealObj, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 2);
	fclose(fp);
	
	fp = fopen (phantom_filename, "rb");
/*	check_error(fp==NULL, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Error in reading file %s\n", phantom_filename);	*/	
	stream_offset = (long int)PHANTOM_OFFSET * (long int)N_z * (long int)N_y * (long int)N_x;
	result = fseek (fp, stream_offset*sizeof(float), SEEK_SET);
/*  	check_error(result != 0, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_filename,stream_offset);*/
	result = fread (&(Init[0][0][0]), sizeof(float), size, fp);
/*  	check_error(result != size, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_filename,result,stream_offset,size);*/
	dwnsmpl_object (ImagObj, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 2);
	fclose(fp);

  	#pragma omp parallel for collapse(3)
        for (slice=0; slice<ScannedObjectPtr->N_z; slice++)
  		for (j=0; j<ScannedObjectPtr->N_y; j++)
      			for (k=0; k<ScannedObjectPtr->N_x; k++)
			{
				if (RealObj[slice][j][k] < 0) RealObj[slice][j][k] = 0;
				else RealObj[slice][j][k] = (ABSORP_COEF_2 - ABSORP_COEF_1)*RealObj[slice][j][k] + ABSORP_COEF_1; 
				if (ImagObj[slice][j][k] < 0) ImagObj[slice][j][k] = 0;
				else ImagObj[slice][j][k] = (REF_IND_DEC_2 - REF_IND_DEC_1)*ImagObj[slice][j][k] + REF_IND_DEC_1;
 			}

	memset(&(RealSino[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
	memset(&(ImagSino[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
  	#pragma omp parallel for private(j, k, p, sino_idx, slice, pixel)
  	for (i=0; i<ScannedObjectPtr->N_time; i++)
  	{
  		for (j=0; j<ScannedObjectPtr->N_y; j++)
    		{
      			for (k=0; k<ScannedObjectPtr->N_x; k++){
        			for (p=0; p<ScannedObjectPtr->ProjNum[i]; p++){
      	    				sino_idx = ScannedObjectPtr->ProjIdxPtr[i][p];
          				calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, SinogramPtr->DetectorResponse, &(AMatrixPtr[i]), j, k, sino_idx);
          				for (slice=0; slice<ScannedObjectPtr->N_z; slice++){
            					pixel = RealObj[slice][j][k]; /*slice+1 to account for extra z slices required for MPI*/
            					forward_project_voxel (SinogramPtr, pixel, RealSino, &(AMatrixPtr[i])/*, &(VoxelLineResponse[slice])*/, sino_idx, slice);
            					pixel = ImagObj[slice][j][k]; /*slice+1 to account for extra z slices required for MPI*/
	        	    			forward_project_voxel (SinogramPtr, pixel, ImagSino, &(AMatrixPtr[i])/*, &(VoxelLineResponse[slice])*/, sino_idx, slice);
          				}
        			}
      			}
    		}
  	}
	
	for (i = 0; i < SinogramPtr->N_p; i++)	
	for (j = 0; j < SinogramPtr->N_r; j++)	
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		SinogramPtr->MagTomoAux[i][j][k][1] = RealSino[i][j][k];	
		SinogramPtr->MagTomoAux[i][j][k][2] += RealSino[i][j][k];	
		SinogramPtr->MagTomoAux[i][j][k][3] += RealSino[i][j][k];
		SinogramPtr->MagTomoAux[i][j][k][0] = (SinogramPtr->MagTomoAux[i][j][k][1] + SinogramPtr->MagTomoAux[i][j][k][2])/2.0;
		
		SinogramPtr->PhaseTomoAux[i][j][k][1] = ImagSino[i][j][k];	
		SinogramPtr->PhaseTomoAux[i][j][k][2] += ImagSino[i][j][k];	
		SinogramPtr->PhaseTomoAux[i][j][k][3] += ImagSino[i][j][k];
		SinogramPtr->PhaseTomoAux[i][j][k][0] = (SinogramPtr->PhaseTomoAux[i][j][k][1] + SinogramPtr->PhaseTomoAux[i][j][k][2])/2.0;

		SinogramPtr->MagPRetAux[i][j][k] = exp(-SinogramPtr->MagTomoAux[i][j][k][1])*cos(-SinogramPtr->PhaseTomoAux[i][j][k][1]);
		SinogramPtr->PhasePRetAux[i][j][k] = exp(-SinogramPtr->MagTomoAux[i][j][k][1])*sin(-SinogramPtr->PhaseTomoAux[i][j][k][1]);
	}
	
	for (i = 0; i < SinogramPtr->N_p; i++)
		compute_phase_projection (SinogramPtr->Measurements_real[i], SinogramPtr->Measurements_imag[i], SinogramPtr->Omega_real[i], SinogramPtr->Omega_imag[i], SinogramPtr->D_real[i], SinogramPtr->D_imag[i], SinogramPtr->MagPRetAux[i], SinogramPtr->PhasePRetAux[i], SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]));
	
	for (i = 0; i < ScannedObjectPtr->N_time; i++)	
	for (j = 0; j < ScannedObjectPtr->N_z; j++)	
	for (k = 0; k < ScannedObjectPtr->N_y; k++)
	for (l = 0; l < ScannedObjectPtr->N_x; l++)
	{
		ScannedObjectPtr->MagObject[i][j+1][k][l] = RealObj[j][k][l];
		ScannedObjectPtr->PhaseObject[i][j+1][k][l] = ImagObj[j][k][l];
	}

    	if (TomoInputsPtr->Write2Tiff == 1)
	  	for (i = 0; i < ScannedObjectPtr->N_time; i++)
	  	{
	    		dimTiff[0] = 1; dimTiff[1] = ScannedObjectPtr->N_z; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;
			sprintf (object_file, "%s_n%d", INIT_MAGOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->MagObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
			sprintf (object_file, "%s_n%d", INIT_PHASEOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->PhaseObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
	  	}
  	
	for (i = 0; i < ScannedObjectPtr->N_time; i++)
  	{
  		free(AMatrixPtr[i].values);
    		free(AMatrixPtr[i].index);
  	}
  
	free (AMatrixPtr);
	multifree(RealObj, 3);	
	multifree(ImagObj, 3);	
	multifree(RealSino, 3);	
	multifree(ImagSino, 3);	
	multifree(Init, 3);	
}

/*'InitObject' intializes the Object to be reconstructed to either 0 or an interpolated version of the previous reconstruction. It is used in multi resolution reconstruction in which after every coarse resolution reconstruction the object should be intialized with an interpolated version of the reconstruction following which the object will be reconstructed at a finer resolution.
--initICD--
If 1, initializes the object to 0
If 2, the code uses bilinear interpolation to initialize the object if the previous reconstruction was at a lower resolution
The function also initializes the magnitude update map 'MagUpdateMap' from the previous coarser resolution
reconstruction. */
int32_t initObject (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  char object_file[100];
  int dimTiff[4];
  int32_t i, j, k, l, size, flag = 0;
  Real_arr_t ***Init, ****UpMapInit;
  
  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  for (j = 0; j < ScannedObjectPtr->N_z; j++)
  for (k = 0; k < ScannedObjectPtr->N_y; k++)
  for (l = 0; l < ScannedObjectPtr->N_x; l++)
  {
  	ScannedObjectPtr->MagObject[i][j+1][k][l] = MAGOBJECT_INIT_VAL;
  	ScannedObjectPtr->PhaseObject[i][j+1][k][l] = PHASEOBJECT_INIT_VAL;
  }

  /*Init = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, PHANTOM_Z_SIZE, PHANTOM_XY_SIZE, PHANTOM_XY_SIZE);
  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  {
	if (read_SharedBinFile_At (, &(ScannedObjectPtr->MagObject[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
 	dwnsmpl_object_bilinear_3D (&(ScannedObjectPtr->MagObject[i][1][0][0]), Init, N_z, N_y, N_x, dwnsmpl_factor);
  }
*/
  if (TomoInputsPtr->initICD > 3 || TomoInputsPtr->initICD < 0){
	sentinel(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: initICD value not recognized.\n");
  }
  else if (TomoInputsPtr->initICD == 1)
  {
	size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
      	for (i = 0; i < ScannedObjectPtr->N_time; i++)
      	{
        	sprintf(object_file, "%s_time_%d", MAGOBJECT_FILENAME,i);
		if (read_SharedBinFile_At (object_file, &(ScannedObjectPtr->MagObject[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
        	sprintf(object_file, "%s_time_%d", PHASEOBJECT_FILENAME,i);
		if (read_SharedBinFile_At (object_file, &(ScannedObjectPtr->PhaseObject[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
      	}
	if (TomoInputsPtr->initMagUpMap == 1)
      	{
		size = ScannedObjectPtr->N_time*TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
		if (read_SharedBinFile_At (UPDATE_MAP_FILENAME, &(ScannedObjectPtr->UpdateMap[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
      	}
  }
  else if (TomoInputsPtr->initICD == 2 || TomoInputsPtr->initICD == 3)
  {
      	if (TomoInputsPtr->initICD == 3)
      	{
        	Init = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
	        check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating object using 3D bilinear interpolation.\n");
	        for (i = 0; i < ScannedObjectPtr->N_time; i++)
        	{
			 size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/8;
         		 sprintf(object_file, "%s_time_%d", MAGOBJECT_FILENAME, i);
			 if (read_SharedBinFile_At (object_file, &(Init[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
          		 upsample_object_bilinear_3D (ScannedObjectPtr->MagObject[i], Init, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
         		 
		 	 sprintf(object_file, "%s_time_%d", PHASEOBJECT_FILENAME, i);
			 if (read_SharedBinFile_At (object_file, &(Init[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
          		 upsample_object_bilinear_3D (ScannedObjectPtr->PhaseObject[i], Init, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
        	}
       		multifree(Init,3);
        	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Done with interpolating object using 3D bilinear interpolation.\n");
      	}
	else
      	{
        	Init = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
	        check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating object using 2D bilinear interpolation.\n");
        	for (i = 0; i < ScannedObjectPtr->N_time; i++)
        	{
	  		size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/4;
         		sprintf(object_file, "%s_time_%d", MAGOBJECT_FILENAME,i);
	  		if (read_SharedBinFile_At (object_file, &(Init[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
          		upsample_object_bilinear_2D (ScannedObjectPtr->MagObject[i], Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
         		sprintf(object_file, "%s_time_%d", PHASEOBJECT_FILENAME,i);
	  		if (read_SharedBinFile_At (object_file, &(Init[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
          		upsample_object_bilinear_2D (ScannedObjectPtr->PhaseObject[i], Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
        	}
        	multifree(Init,3);
        	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Done with interpolating object using 2D bilinear interpolation.\n");
      	}
        if (TomoInputsPtr->initMagUpMap == 1)
        {
          	if (TomoInputsPtr->prevnum_z_blocks == TomoInputsPtr->num_z_blocks)
          	{	
			UpMapInit = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
			size = ScannedObjectPtr->N_time*TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/4;
			if (read_SharedBinFile_At (UPDATE_MAP_FILENAME, &(UpMapInit[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
          		check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating magnitude update map using 2D bilinear interpolation.\n");
          		upsample_bilinear_2D (ScannedObjectPtr->UpdateMap, UpMapInit, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
          		multifree(UpMapInit,4);
	  	}
		else if (TomoInputsPtr->prevnum_z_blocks == TomoInputsPtr->num_z_blocks/2)
	  	{
			UpMapInit = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
			size = ScannedObjectPtr->N_time*TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/8;
			if (read_SharedBinFile_At (UPDATE_MAP_FILENAME, &(UpMapInit[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
          		check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating magnitude update map using 3D bilinear interpolation.\n");
			upsample_bilinear_3D (ScannedObjectPtr->UpdateMap, UpMapInit, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
          		multifree(UpMapInit,4);
	  	}
	  	else
	  	{
			check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of axial blocks is incompatible with previous stage of multi-resolution.\n");
			check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Initializing the multi-resolution map to zeros.\n");
	  	}	
          }
      }
  
  	dimTiff[0] = ScannedObjectPtr->N_time; dimTiff[1] = TomoInputsPtr->num_z_blocks; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;
  	sprintf(object_file, "%s_n%d", UPDATE_MAP_FILENAME, TomoInputsPtr->node_rank);
  	if (TomoInputsPtr->Write2Tiff == 1)
  		if (WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->UpdateMap[0][0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr))
			flag = -1;
  
    	if (TomoInputsPtr->Write2Tiff == 1)
	  	for (i = 0; i < ScannedObjectPtr->N_time; i++)
	  	{
	    		dimTiff[0] = 1; dimTiff[1] = ScannedObjectPtr->N_z; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;
			sprintf (object_file, "%s_n%d", INIT_MAGOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			if (WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->MagObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr))flag = -1;
			sprintf (object_file, "%s_n%d", INIT_PHASEOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			if (WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->PhaseObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr))flag = -1;
	  	}
	
	return (flag);
error:
	return (-1);
}


/*'initErrorSinogram' is used to initialize the error sinogram before start of ICD. It computes e = y - Ax - d. Ax is computed by forward projecting the obkject x.*/
int32_t initErrorSinogam (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr/*, AMatrixCol* VoxelLineResponse*/)
{
  Real_arr_t*** MagErrorSino = SinogramPtr->MagErrorSino;
  Real_arr_t*** PhaseErrorSino = SinogramPtr->PhaseErrorSino;
  Real_t pixel, magavg = 0, phaseavg = 0;
  int32_t dimTiff[4], i, j, k, p, sino_idx, slice, flag = 0;
  AMatrixCol* AMatrixPtr = (AMatrixCol*)get_spc(ScannedObjectPtr->N_time, sizeof(AMatrixCol));
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
  char error_file[100];

  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  {
    AMatrixPtr[i].values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
    AMatrixPtr[i].index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  }
  memset(&(MagErrorSino[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
  memset(&(PhaseErrorSino[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
  #pragma omp parallel for private(j, k, p, sino_idx, slice, pixel)
  for (i=0; i<ScannedObjectPtr->N_time; i++)
  {
    for (j=0; j<ScannedObjectPtr->N_y; j++)
    {
      for (k=0; k<ScannedObjectPtr->N_x; k++){
        for (p=0; p<ScannedObjectPtr->ProjNum[i]; p++){
          sino_idx = ScannedObjectPtr->ProjIdxPtr[i][p];
          calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, SinogramPtr->DetectorResponse, &(AMatrixPtr[i]), j, k, sino_idx);
          for (slice=0; slice<ScannedObjectPtr->N_z; slice++){
            /*	printf("count = %d, idx = %d, val = %f\n", VoxelLineResponse[slice].count, VoxelLineResponse[slice].index[0], VoxelLineResponse[slice].values[0]);*/
            pixel = ScannedObjectPtr->MagObject[i][slice+1][j][k]; /*slice+1 to account for extra z slices required for MPI*/
            forward_project_voxel (SinogramPtr, pixel, MagErrorSino, &(AMatrixPtr[i])/*, &(VoxelLineResponse[slice])*/, sino_idx, slice);
            pixel = ScannedObjectPtr->PhaseObject[i][slice+1][j][k]; /*slice+1 to account for extra z slices required for MPI*/
            forward_project_voxel (SinogramPtr, pixel, PhaseErrorSino, &(AMatrixPtr[i])/*, &(VoxelLineResponse[slice])*/, sino_idx, slice);
          }
        }
      }
    }
  }
  
  #pragma omp parallel for private(j, k) reduction(+:magavg,phaseavg)
  for(i = 0; i < SinogramPtr->N_p; i++)
  for (j = 0; j < SinogramPtr->N_r; j++)
  for (k = 0; k < SinogramPtr->N_t; k++)
  {
    MagErrorSino[i][j][k] = SinogramPtr->MagTomoAux[i][j][k][1] - SinogramPtr->MagTomoDual[i][j][k] - MagErrorSino[i][j][k];
    PhaseErrorSino[i][j][k] = SinogramPtr->PhaseTomoAux[i][j][k][1] - SinogramPtr->PhaseTomoDual[i][j][k] - PhaseErrorSino[i][j][k];
    magavg += MagErrorSino[i][j][k];
    phaseavg += PhaseErrorSino[i][j][k];
  }
  magavg = magavg/(SinogramPtr->N_r*SinogramPtr->N_t*SinogramPtr->N_p);
  phaseavg = phaseavg/(SinogramPtr->N_r*SinogramPtr->N_t*SinogramPtr->N_p);
  check_debug(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "Average of magnitude and phase components of error sinogram in node %d are %f and %f\n", TomoInputsPtr->node_rank, magavg, phaseavg);
  
  dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
  if (TomoInputsPtr->Write2Tiff == 1)
  {
  	sprintf(error_file, "%s_n%d", MAGTOMOAUX_FILENAME, TomoInputsPtr->node_rank);
  	flag = WriteMultiDimArray2Tiff (error_file, dimTiff, 0, 3, 1, 2, &(SinogramPtr->MagTomoAux[0][0][0][0]), 0, 1, 4, TomoInputsPtr->debug_file_ptr);
  	sprintf(error_file, "%s_n%d", PHASETOMOAUX_FILENAME, TomoInputsPtr->node_rank);
  	flag |= WriteMultiDimArray2Tiff (error_file, dimTiff, 0, 3, 1, 2, &(SinogramPtr->PhaseTomoAux[0][0][0][0]), 0, 1, 4, TomoInputsPtr->debug_file_ptr);
  	sprintf(error_file, "%s_n%d", MAGTOMODUAL_FILENAME, TomoInputsPtr->node_rank);
  	flag = WriteMultiDimArray2Tiff (error_file, dimTiff, 0, 3, 1, 2, &(SinogramPtr->MagTomoDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  	sprintf(error_file, "%s_n%d", PHASETOMODUAL_FILENAME, TomoInputsPtr->node_rank);
  	flag |= WriteMultiDimArray2Tiff (error_file, dimTiff, 0, 3, 1, 2, &(SinogramPtr->PhaseTomoDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  }

  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  {
    free(AMatrixPtr[i].values);
    free(AMatrixPtr[i].index);
  }
  free (AMatrixPtr);
  return (flag);
}


  /*Implements mutithreaded shared memory parallelization using OpenMP and splits work among
  threads. Each thread gets a certain time slice and z block to update.
  Multithreading is done within the z-blocks assigned to each node.
  ErrorSino - Error sinogram
  Iter - Present iteration number
  MagUpdateMap - Magnitude update map containing the magnitude of update of each voxel
  Mask - If a certain element is true then the corresponding voxel is updated*/
  int updateVoxelsTimeSlices(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, int32_t Iter, uint8_t*** Mask)
  {
    Real_t AverageUpdate = 0, tempUpdate, avg_update_percentage, total_vox_mag = 0.0, vox_mag = 0.0;
    int32_t xy_start, xy_end, i, j, K, block, idx, **z_start, **z_stop;
    Real_t tempTotPix = 0, total_pix = 0;
    long int **zero_count, total_zero_count = 0;
    int32_t** thread_num = (int32_t**)multialloc(sizeof(int32_t), 2, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks);
    MPI_Request *mag_send_reqs, *mag_recv_reqs, *phase_send_reqs, *phase_recv_reqs;
    mag_send_reqs = (MPI_Request*)get_spc(ScannedObjectPtr->N_time, sizeof(MPI_Request));
    mag_recv_reqs = (MPI_Request*)get_spc(ScannedObjectPtr->N_time, sizeof(MPI_Request));
    phase_send_reqs = (MPI_Request*)get_spc(ScannedObjectPtr->N_time, sizeof(MPI_Request));
    phase_recv_reqs = (MPI_Request*)get_spc(ScannedObjectPtr->N_time, sizeof(MPI_Request));
    z_start = (int32_t**)multialloc(sizeof(int32_t), 2, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks);
    z_stop = (int32_t**)multialloc(sizeof(int32_t), 2, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks);
    
    randomly_select_x_y (ScannedObjectPtr, TomoInputsPtr, Mask);
    
    zero_count = (long int**)multialloc(sizeof(long int), 2, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks);
    
    memset(&(zero_count[0][0]), 0, ScannedObjectPtr->N_time*TomoInputsPtr->num_z_blocks*sizeof(long int));
    /*	K = ScannedObjectPtr->N_time*ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
    K = (K - total_zero_count)/(ScannedObjectPtr->gamma*K);*/
    K = ScannedObjectPtr->NHICD_Iterations;
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of NHICD iterations is %d.\n", K);
    for (j = 0; j < K; j++)
    {
      total_vox_mag = 0.0;
      #pragma omp parallel for collapse(2) private(i, block, idx, xy_start, xy_end) reduction(+:total_vox_mag)
      for (i = 0; i < ScannedObjectPtr->N_time; i++)
      for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
      {
        idx = (i % 2 == 0) ? block: block + 1;
        z_start[i][idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
        z_stop[i][idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
        z_stop[i][idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[i][idx];
        xy_start = j*floor(TomoInputsPtr->UpdateSelectNum[i][idx]/K);
        xy_end = (j + 1)*floor(TomoInputsPtr->UpdateSelectNum[i][idx]/K) - 1;
        xy_end = (j == K - 1) ? TomoInputsPtr->UpdateSelectNum[i][idx] - 1: xy_end;
        /*	printf ("Loop 1 Start - j = %d, i = %d, idx = %d, z_start = %d, z_stop = %d, xy_start = %d, xy_end = %d\n", j, i, idx, z_start[i][idx], z_stop[i][idx], xy_start, xy_end);*/
        total_vox_mag += updateVoxels (i, i, z_start[i][idx], z_stop[i][idx], xy_start, xy_end, TomoInputsPtr->x_rand_select[i][idx], TomoInputsPtr->y_rand_select[i][idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->MagErrorSino, SinogramPtr->PhaseErrorSino, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[i][idx]), ScannedObjectPtr->UpdateMap[i][idx], Mask[i]);
        thread_num[i][idx] = omp_get_thread_num();
      }
      
      /*check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Send MPI info\n");*/
      MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 0);
      /*	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "update_Sinogram_Offset: Will compute projection offset error\n");*/
      MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 0);
      #pragma omp parallel for collapse(2) private(i, block, idx, xy_start, xy_end) reduction(+:total_vox_mag)
      for (i = 0; i < ScannedObjectPtr->N_time; i++)
      for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
      {
        idx = (i % 2 == 0) ? block + 1: block;
        z_start[i][idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
        z_stop[i][idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
        z_stop[i][idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[i][idx];
        xy_start = j*floor(TomoInputsPtr->UpdateSelectNum[i][idx]/K);
        xy_end = (j + 1)*floor(TomoInputsPtr->UpdateSelectNum[i][idx]/K) - 1;
        xy_end = (j == K - 1) ? TomoInputsPtr->UpdateSelectNum[i][idx] - 1: xy_end;
        total_vox_mag += updateVoxels (i, i, z_start[i][idx], z_stop[i][idx], xy_start, xy_end, TomoInputsPtr->x_rand_select[i][idx], TomoInputsPtr->y_rand_select[i][idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->MagErrorSino, SinogramPtr->PhaseErrorSino, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[i][idx]), ScannedObjectPtr->UpdateMap[i][idx], Mask[i]);
        thread_num[i][idx] = omp_get_thread_num();
        /*	printf ("Loop 2 - i = %d, idx = %d, z_start = %d, z_stop = %d, xy_start = %d, xy_end = %d\n", i, idx, z_start[i][idx], z_stop[i][idx], xy_start, xy_end);*/
      }
      
      MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 1);
      MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 1);
      VSC_based_Voxel_Line_Select(ScannedObjectPtr, TomoInputsPtr, ScannedObjectPtr->UpdateMap);
      /*	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of NHICD voxel lines to be updated in iteration %d is %d\n", j, num_voxel_lines);*/
      if (Iter > 1 && TomoInputsPtr->no_NHICD == 0)
      {
        #pragma omp parallel for collapse(2) private(i, block, idx)
        for (i = 0; i < ScannedObjectPtr->N_time; i++)
        for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
        {
          idx = (i % 2 == 0) ? block: block + 1;
          z_start[i][idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
          z_stop[i][idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
          z_stop[i][idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[i][idx];
          updateVoxels (i, i, z_start[i][idx], z_stop[i][idx], 0, TomoInputsPtr->NHICDSelectNum[i][idx]-1, TomoInputsPtr->x_NHICD_select[i][idx], TomoInputsPtr->y_NHICD_select[i][idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->MagErrorSino, SinogramPtr->PhaseErrorSino, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[i][idx]), ScannedObjectPtr->UpdateMap[i][idx], Mask[i]);
          thread_num[i][idx] = omp_get_thread_num();
          /*	printf ("Loop 1 NHICD - i = %d, idx = %d, z_start = %d, z_stop = %d\n", i, idx, z_start[i][idx], z_stop[i][idx]);*/
        }
        
        MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 0);
        MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 0);
        
        #pragma omp parallel for collapse(2) private(i, block, idx)
        for (i = 0; i < ScannedObjectPtr->N_time; i++)
        for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
        {
          idx = (i % 2 == 0) ? block + 1: block;
          z_start[i][idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
          z_stop[i][idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
          z_stop[i][idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[i][idx];
          updateVoxels (i, i, z_start[i][idx], z_stop[i][idx], 0, TomoInputsPtr->NHICDSelectNum[i][idx]-1, TomoInputsPtr->x_NHICD_select[i][idx], TomoInputsPtr->y_NHICD_select[i][idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->MagErrorSino, SinogramPtr->PhaseErrorSino, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[i][idx]), ScannedObjectPtr->UpdateMap[i][idx], Mask[i]);
          thread_num[i][idx] = omp_get_thread_num();
          /*	printf ("Loop 2 NHICD - i = %d, idx = %d, z_start = %d, z_stop = %d\n", i, idx, z_start[i][idx], z_stop[i][idx]);*/
        }
        
        MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 1);
        MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, mag_send_reqs, phase_send_reqs, mag_recv_reqs, phase_recv_reqs, 1);
      }
    }
    
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Time Slice, Z Start, Z End - Thread : ");
    total_pix = 0;
    for (i=0; i<ScannedObjectPtr->N_time; i++){
      for (block=0; block<TomoInputsPtr->num_z_blocks; block++){
        total_pix += TomoInputsPtr->UpdateSelectNum[i][block]*(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
        for (j=0; j<TomoInputsPtr->UpdateSelectNum[i][block]; j++){
          AverageUpdate += ScannedObjectPtr->UpdateMap[i][block][TomoInputsPtr->y_rand_select[i][block][j]][TomoInputsPtr->x_rand_select[i][block][j]];
        }
        total_zero_count += zero_count[i][block];
        check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "%d,%d,%d-%d; ", i, z_start[i][block], z_stop[i][block], thread_num[i][block]);
      }
    }
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "\n");
    
    MPI_Allreduce(&AverageUpdate, &tempUpdate, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_pix, &tempTotPix, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_vox_mag, &vox_mag, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    AverageUpdate = tempUpdate/(tempTotPix);
    AverageUpdate = convert2Hounsfield(AverageUpdate);
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Average voxel update over all voxels is %f, total voxels is %f.\n", AverageUpdate, tempTotPix);
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Zero count is %ld.\n", total_zero_count);
    
    multifree(zero_count,2);
    multifree(thread_num,2);
    multifree(z_start,2);
    multifree(z_stop,2);
    free(mag_send_reqs);
    free(mag_recv_reqs);
    free(phase_send_reqs);
    free(phase_recv_reqs);
    /*	multifree(offset_numerator,2);
    multifree(offset_denominator,2);*/
    avg_update_percentage = 100*tempUpdate/vox_mag;
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Percentage average magnitude of voxel updates is %f.\n", avg_update_percentage);
    
    if (avg_update_percentage < TomoInputsPtr->StopThreshold)
    {
      check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Percentage average magnitude of voxel updates is less than convergence threshold.\n");
      return (1);
    }
    return(0);
  }

  /*ICD_BackProject calls the ICD optimization function repeatedly till the stopping criteria is met.*/
  int ICD_BackProject(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
  {
    #ifndef NO_COST_CALCULATE
    Real_t cost, cost_0_iter, cost_last_iter, percentage_change_in_cost = 0;
    char costfile[100] = COST_FILENAME, origcostfile[100] = ORIG_COST_FILENAME;
    #endif
    Real_t x, y, ar, ai, orig_cost, orig_cost_last;
    int32_t j, flag = 0, Iter, i, k, HeadIter;
    int dimTiff[4];
    time_t start;
    char detect_file[100] = DETECTOR_RESPONSE_FILENAME;
    char MagUpdateMapFile[100] = UPDATE_MAP_FILENAME, aux_filename[100];
    uint8_t ***Mask;
    Real_arr_t*** omega_abs = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

    /*AMatrixCol *VoxelLineResponse;*/
    #ifdef POSITIVITY_CONSTRAINT
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Enforcing positivity constraint\n");
    #endif
    
    ScannedObjectPtr->UpdateMap = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
    SinogramPtr->DetectorResponse = (Real_arr_t **)multialloc(sizeof(Real_arr_t), 2, SinogramPtr->N_p, DETECTOR_RESPONSE_BINS + 1);
    SinogramPtr->MagErrorSino = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
    SinogramPtr->PhaseErrorSino = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
    Mask = (uint8_t***)multialloc(sizeof(uint8_t), 3, ScannedObjectPtr->N_time, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
    
    memset(&(ScannedObjectPtr->UpdateMap[0][0][0][0]), 0, ScannedObjectPtr->N_time*TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x*sizeof(Real_arr_t));
/*    omp_set_num_threads(TomoInputsPtr->num_threads);*/
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of CPU cores is %d\n", (int)omp_get_num_procs());
    /*	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ICD_BackProject: Number of threads is %d\n", TomoInputsPtr->num_threads) ;*/
    for (i = 0; i < ScannedObjectPtr->N_time; i++)
    for (j = 0; j < ScannedObjectPtr->N_y; j++)
    for (k = 0; k < ScannedObjectPtr->N_x; k++){
      x = ScannedObjectPtr->x0 + ((Real_t)k + 0.5)*ScannedObjectPtr->delta_xy;
      y = ScannedObjectPtr->y0 + ((Real_t)j + 0.5)*ScannedObjectPtr->delta_xy;
      if (x*x + y*y < TomoInputsPtr->radius_obj*TomoInputsPtr->radius_obj)
        Mask[i][j][k] = 1;
      else
        Mask[i][j][k] = 0;
    }
    
    SinogramPtr->fftforw_arr = (fftw_complex**)get_spc(SinogramPtr->N_p, sizeof(fftw_complex*));
    SinogramPtr->fftback_arr = (fftw_complex**)get_spc(SinogramPtr->N_p, sizeof(fftw_complex*));
    SinogramPtr->fftforw_plan = (fftw_plan*)get_spc(SinogramPtr->N_p, sizeof(fftw_plan));
    SinogramPtr->fftback_plan = (fftw_plan*)get_spc(SinogramPtr->N_p, sizeof(fftw_plan));
    for (i = 0; i < SinogramPtr->N_p; i++)
    {
	SinogramPtr->fftforw_arr[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*SinogramPtr->N_t*SinogramPtr->N_r);
	SinogramPtr->fftback_arr[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*SinogramPtr->N_t*SinogramPtr->N_r);
	SinogramPtr->fftforw_plan[i] = fftw_plan_dft_2d(SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->fftforw_arr[i], SinogramPtr->fftforw_arr[i], FFTW_FORWARD, FFTW_ESTIMATE);
	SinogramPtr->fftback_plan[i] = fftw_plan_dft_2d(SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->fftback_arr[i], SinogramPtr->fftback_arr[i], FFTW_BACKWARD, FFTW_ESTIMATE);
    }
 
    DetectorResponseProfile (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
    dimTiff[0] = 1; dimTiff[1] = 1; dimTiff[2] = SinogramPtr->N_p; dimTiff[3] = DETECTOR_RESPONSE_BINS+1;
    sprintf(detect_file, "%s_n%d", detect_file, TomoInputsPtr->node_rank);
    if (TomoInputsPtr->Write2Tiff == 1)
    	if (WriteMultiDimArray2Tiff (detect_file, dimTiff, 0, 1, 2, 3, &(SinogramPtr->DetectorResponse[0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    start = time(NULL);
    
    if (initObject(SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;

    if (TomoInputsPtr->initICD == 0)  
	    init_GroundTruth (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
    
    if (init_minmax_object (ScannedObjectPtr, TomoInputsPtr)) goto error;
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Time taken to read object = %fmins\n", difftime(time(NULL),start)/60.0);
    if (initErrorSinogam(SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;

    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Time taken to initialize object and compute error sinogram = %fmins\n", difftime(time(NULL),start)/60.0);
  
    start=time(NULL);
    orig_cost_last = compute_original_cost(SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
    check_info(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "HeadIter = 0: The original cost value is %f. \n", orig_cost_last);
    if (TomoInputsPtr->node_rank == 0)
	   Write2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost_last, TomoInputsPtr->debug_file_ptr);

    for (HeadIter = 1; HeadIter <= TomoInputsPtr->Head_MaxIter; HeadIter++)
    {
    	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Doing phase retrieval ....\n");

        #pragma omp parallel for private(j,k)
	for (i = 0; i < SinogramPtr->N_p; i++)
	{
		for (j = 0; j < SinogramPtr->N_r; j++)
		for (k = 0; k < SinogramPtr->N_t; k++)
		{
			SinogramPtr->MagErrorSino[i][j][k] = -(SinogramPtr->MagErrorSino[i][j][k] - SinogramPtr->MagTomoAux[i][j][k][1]);
			SinogramPtr->PhaseErrorSino[i][j][k] = -(SinogramPtr->PhaseErrorSino[i][j][k] - SinogramPtr->PhaseTomoAux[i][j][k][1]);
		}

		estimate_complex_projection (SinogramPtr->Measurements_real[i], SinogramPtr->Measurements_imag[i], SinogramPtr->Omega_real[i], SinogramPtr->Omega_imag[i], SinogramPtr->D_real[i], SinogramPtr->D_imag[i], SinogramPtr->MagTomoAux[i], SinogramPtr->PhaseTomoAux[i], TomoInputsPtr->Weight[i], SinogramPtr->MagErrorSino[i], SinogramPtr->PhaseErrorSino[i], SinogramPtr->MagPRetAux[i], SinogramPtr->PhasePRetAux[i], SinogramPtr->MagPRetDual[i], SinogramPtr->PhasePRetDual[i], SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, TomoInputsPtr->NMS_rho, TomoInputsPtr->NMS_chi, TomoInputsPtr->NMS_gamma, TomoInputsPtr->NMS_sigma, TomoInputsPtr->NMS_threshold, TomoInputsPtr->NMS_MaxIter, TomoInputsPtr->SteepDes_threshold, TomoInputsPtr->SteepDes_MaxIter, TomoInputsPtr->PRet_threshold, TomoInputsPtr->PRet_MaxIter, TomoInputsPtr->ADMM_mu, TomoInputsPtr->ADMM_mu, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]));
	
		for (j = 0; j < SinogramPtr->N_r; j++)
		for (k = 0; k < SinogramPtr->N_t; k++)
		{
			SinogramPtr->MagErrorSino[i][j][k] = SinogramPtr->MagTomoAux[i][j][k][1] - SinogramPtr->MagErrorSino[i][j][k];
			SinogramPtr->PhaseErrorSino[i][j][k] = SinogramPtr->PhaseTomoAux[i][j][k][1] - SinogramPtr->PhaseErrorSino[i][j][k];
		}
	}

  	dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
  	if (TomoInputsPtr->Write2Tiff == 1)
  	{
  		sprintf(aux_filename, "%s_n%d", MAGTOMOAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->MagTomoAux[0][0][0][0]), 0, 1, 4, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASETOMOAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->PhaseTomoAux[0][0][0][0]), 0, 1, 4, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", MAGTOMODUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->MagTomoDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASETOMODUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->PhaseTomoDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		
		sprintf(aux_filename, "%s_n%d", MAGPRETAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->MagPRetAux[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASEPRETAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->PhasePRetAux[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", MAGPRETDUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->MagPRetDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASEPRETDUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(SinogramPtr->PhasePRetDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);

		for (i = 0; i < SinogramPtr->N_p; i++)
		for (j = 0; j < SinogramPtr->N_r; j++)
		for (k = 0; k < SinogramPtr->N_t; k++)
			omega_abs[i][j][k] = sqrt(SinogramPtr->Omega_real[i][j][k]*SinogramPtr->Omega_real[i][j][k] + SinogramPtr->Omega_imag[i][j][k]*SinogramPtr->Omega_imag[i][j][k]);
  		
		sprintf(aux_filename, "%s_n%d", OMEGAABS_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 3, 1, 2, &(omega_abs[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  	}

#ifndef NO_COST_CALCULATE
	    cost = computeCost(SinogramPtr,ScannedObjectPtr,TomoInputsPtr);
	    cost_0_iter = cost;
	    cost_last_iter = cost;
	    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "------------- Iteration 0, Cost = %f------------\n",cost);
	    if (TomoInputsPtr->node_rank == 0)
	    	Write2Bin (costfile, 1, 1, 1, 1, sizeof(Real_t), &cost, TomoInputsPtr->debug_file_ptr);
#endif /*Cost calculation endif*/
    	for (Iter = 1; Iter <= TomoInputsPtr->NumIter; Iter++)
    	{
      		flag = updateVoxelsTimeSlices (SinogramPtr, ScannedObjectPtr, TomoInputsPtr, Iter, Mask);
      		if (TomoInputsPtr->WritePerIter == 1)
      			if (write_ObjectProjOff2TiffBinPerIter (SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;
#ifndef NO_COST_CALCULATE
	      cost = computeCost(SinogramPtr,ScannedObjectPtr,TomoInputsPtr);
	      percentage_change_in_cost = ((cost - cost_last_iter)/(cost - cost_0_iter))*100.0;
	      check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Percentage change in cost is %f.\n", percentage_change_in_cost);
	      check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "------------- Iteration = %d, Cost = %f, Time since start of ICD = %fmins ------------\n",Iter,cost,difftime(time(NULL),start)/60.0);
	      if (TomoInputsPtr->node_rank == 0)
			Append2Bin (costfile, 1, 1, 1, 1, sizeof(Real_t), &cost, TomoInputsPtr->debug_file_ptr);
	      
	      if (cost > cost_last_iter)
		      check_warn(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "Cost value increased.\n");
	      cost_last_iter = cost;
	      /*if (percentage_change_in_cost < TomoInputsPtr->cost_thresh && flag != 0 && Iter > 1){*/
	      if (flag != 0 && Iter > 1){
		        check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Convergence criteria is met.\n");
        		break;
      		}
#else
	      check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "-------------ICD_BackProject: ICD Iter = %d, time since start of ICD = %fmins------------.\n",Iter,difftime(time(NULL),start)/60.0);
		if (flag != 0 && Iter > 1){
        		check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Convergence criteria is met.\n");
        		break;
	      }
#endif
	      flag = fflush(TomoInputsPtr->debug_file_ptr);
     		 if (flag != 0)
      			check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cannot flush buffer.\n");
    	}

        #pragma omp parallel for collapse(3) private(i,j,k,ar,ai)
	for (i = 0; i < SinogramPtr->N_p; i++)
	for (j = 0; j < SinogramPtr->N_r; j++)
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		ar = SinogramPtr->MagTomoDual[i][j][k]; ai = SinogramPtr->PhaseTomoDual[i][j][k];
		SinogramPtr->MagTomoDual[i][j][k] = -SinogramPtr->MagErrorSino[i][j][k];
		SinogramPtr->PhaseTomoDual[i][j][k] = -SinogramPtr->PhaseErrorSino[i][j][k];
		SinogramPtr->MagErrorSino[i][j][k] += ar - SinogramPtr->MagTomoDual[i][j][k];
		SinogramPtr->PhaseErrorSino[i][j][k] += ai - SinogramPtr->PhaseTomoDual[i][j][k];
    	}
	
	orig_cost = compute_original_cost(SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
        check_info(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "HeadIter = %d: The original cost value is %f. The decrease in original cost is %f.\n", HeadIter, orig_cost, orig_cost_last - orig_cost);
    	if (TomoInputsPtr->node_rank == 0)
	   Write2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost, TomoInputsPtr->debug_file_ptr);
	
	if (orig_cost > orig_cost_last)
      		check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cost of original cost function increased!\n");
	orig_cost_last = orig_cost;
    }   

    for (i = 0; i < SinogramPtr->N_p; i++)
    { 
	fftw_destroy_plan(SinogramPtr->fftforw_plan[i]);
	fftw_destroy_plan(SinogramPtr->fftback_plan[i]);
        fftw_free(SinogramPtr->fftforw_arr[i]); 
	fftw_free(SinogramPtr->fftback_arr[i]);
    }
    free(SinogramPtr->fftforw_arr);
    free(SinogramPtr->fftback_arr);
    
    int32_t size = ScannedObjectPtr->N_time*TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
    if (write_SharedBinFile_At (MagUpdateMapFile, &(ScannedObjectPtr->UpdateMap[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
    
	size = SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t*4;
	if (write_SharedBinFile_At (MAGTOMOAUX_FILENAME, &(SinogramPtr->MagTomoAux[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error; 
	if (write_SharedBinFile_At (PHASETOMOAUX_FILENAME, &(SinogramPtr->PhaseTomoAux[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
	size = SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t;
	if (write_SharedBinFile_At (MAGPRETAUX_FILENAME, &(SinogramPtr->MagPRetAux[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error; 
	if (write_SharedBinFile_At (PHASEPRETAUX_FILENAME, &(SinogramPtr->PhasePRetAux[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
	if (write_SharedBinFile_At (OMEGAREAL_FILENAME, &(SinogramPtr->Omega_real[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
	if (write_SharedBinFile_At (OMEGAIMAG_FILENAME, &(SinogramPtr->Omega_imag[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
 
    multifree(SinogramPtr->MagErrorSino,3);
    multifree(SinogramPtr->PhaseErrorSino,3);
    multifree(SinogramPtr->DetectorResponse,2);
    multifree(Mask,3);
    multifree(ScannedObjectPtr->UpdateMap, 4);
    multifree(omega_abs, 2);
 
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Finished running ICD_BackProject.\n");
    flag = fflush(TomoInputsPtr->debug_file_ptr);
    if (flag != 0)
       check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cannot flush buffer.\n");
    
    return(0);

error:
    multifree(SinogramPtr->MagErrorSino,3);
    multifree(SinogramPtr->PhaseErrorSino,3);
    multifree(SinogramPtr->DetectorResponse,2);
    multifree(Mask,3);
    multifree(ScannedObjectPtr->UpdateMap, 4);
    return(-1);
  }

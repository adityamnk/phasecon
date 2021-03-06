/* ===========================================================================
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
#include "XT_Paganin.h"
#include "XT_ObjectInit.h"

int32_t initErrorSinogam(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
int updateVoxelsTimeSlices(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, int32_t Iter, uint8_t*** Mask);

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
  Real_t cost=0, temp=0, forward=0, prior=0;
  Real_t Diff_delta, Diff_beta, Diff;
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
  #pragma omp parallel for private(Diff_delta, Diff_beta, Diff, p, j, k, j_minus, k_minus, p_plus, i_plus, j_plus, k_plus) reduction(+:temp)
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
        Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j][k + 1]);
        Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j][k + 1]);
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
      }
      if(j_plus == true) {
        if(k_minus == true) {
          Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k - 1]);
          Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k - 1]);
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
        }
        Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k]);
        Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k]);
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
        if(k_plus == true) {
          Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k + 1]);
          Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k + 1]);
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
        }
      }
      if (p_plus == true)
      {
        if(j_minus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
        
        Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j][k];
        Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j][k];
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        if(j_plus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j + 1][k];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j + 1][k];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_minus == true)
        {
          if(k_minus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k - 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k - 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k + 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k + 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_minus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k - 1];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k - 1];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_plus == true)
        {
          if(k_minus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k - 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k - 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k + 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k + 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_plus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k + 1];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k + 1];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
      }
      if(i_plus == true) {
        Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i+1][p][j][k]);
        Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i+1][p][j][k]);
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Time_Filter[0] * Phase_QGGMRF_Temporal_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Time_Filter[0] * Mag_QGGMRF_Temporal_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
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
  Real_t cost=0,temp=0, forward=0, prior=0, Diff_delta, Diff_beta, Diff, magtemp, costemp, sintemp; 
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
	compute_FresnelTran (SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]), SinogramPtr->Light_Wavelength, SinogramPtr->Obj2Det_Distance, SinogramPtr->Freq_Window);
		
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
  #pragma omp parallel for private(Diff_delta, Diff_beta, Diff, p, j, k, j_minus, k_minus, p_plus, i_plus, j_plus, k_plus) reduction(+:temp)
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
        Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j][k + 1]);
        Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j][k + 1]);
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
      }
      if(j_plus == true) {
        if(k_minus == true) {
          Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k - 1]);
          Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k - 1]);
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
        }
        Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k]);
        Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k]);
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
        if(k_plus == true) {
          Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p][j + 1][k + 1]);
          Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p][j + 1][k + 1]);
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Phase_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * Mag_QGGMRF_Spatial_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
        }
      }
      if (p_plus == true)
      {
        if(j_minus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
        
        Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j][k];
        Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j][k];
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        if(j_plus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p+1][j + 1][k];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p+1][j + 1][k];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_minus == true)
        {
          if(k_minus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k - 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k - 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j - 1][k + 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j - 1][k + 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_minus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k - 1];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k - 1];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
        if(j_plus == true)
        {
          if(k_minus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k - 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k - 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
          if(k_plus == true)
          {
            Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j + 1][k + 1];
            Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j + 1][k + 1];
	    Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	    Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
          }
        }
        if(k_plus == true)
        {
          Diff_delta = ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i][p + 1][j][k + 1];
          Diff_beta = ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i][p + 1][j][k + 1];
	  Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Phase_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
	  Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * Mag_QGGMRF_Spatial_Value(Diff, ScannedObjectPtr, TomoInputsPtr);
        }
      }
      if(i_plus == true) {
        Diff_delta = (ScannedObjectPtr->PhaseObject[i][p][j][k] - ScannedObjectPtr->PhaseObject[i+1][p][j][k]);
        Diff_beta = (ScannedObjectPtr->MagObject[i][p][j][k] - ScannedObjectPtr->MagObject[i+1][p][j][k]);
	Diff = ScannedObjectPtr->DecorrTran[0][0]*Diff_delta + ScannedObjectPtr->DecorrTran[0][1]*Diff_beta;
        temp += TomoInputsPtr->Time_Filter[0] * Phase_QGGMRF_Temporal_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
	Diff = ScannedObjectPtr->DecorrTran[1][0]*Diff_delta + ScannedObjectPtr->DecorrTran[1][1]*Diff_beta;
        temp += TomoInputsPtr->Time_Filter[0] * Mag_QGGMRF_Temporal_Value(Diff,ScannedObjectPtr,TomoInputsPtr);
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


int do_PagPhaseRet_MBIRRecon (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, uint8_t*** Mask)
{
	Real_arr_t ***z_real, ***z_imag, ***ProjLength; float ***Init; char object_file[100];
	Real_t cost, cost_last_iter, cost_0_iter, percentage_change_in_cost; 
	int32_t i, j, k, dimTiff[4], Iter, flag;
	int64_t size;
  	AMatrixCol* AMatrixPtr = (AMatrixCol*)get_spc(ScannedObjectPtr->N_time, sizeof(AMatrixCol));
  	uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
 
	Init = (float***)multialloc(sizeof(float), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
  	ProjLength = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	z_real = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	z_imag = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	
	for (i = 0; i < ScannedObjectPtr->N_time; i++)
	{
    		AMatrixPtr[i].values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
    		AMatrixPtr[i].index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  	}

	for (i = 0; i < SinogramPtr->N_p; i++)	
	{
		printf("projection index  i = %d\n", i);
/*		paganins_2mat_phase_retrieval (SinogramPtr->Measurements_real[i], SinogramPtr->D_real[i], SinogramPtr->D_imag[i], ProjLength[i], z_real[i], z_imag[i], SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]), SinogramPtr->Light_Wavenumber, SinogramPtr->Light_Wavelength, SinogramPtr->Obj2Det_Distance, SinogramPtr->Delta_Over_Beta);*/
		paganins_1mat_phase_retrieval (SinogramPtr->Measurements_real[i], SinogramPtr->D_real[i], SinogramPtr->D_imag[i], z_real[i], z_imag[i], SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]), SinogramPtr->Light_Wavenumber, SinogramPtr->Light_Wavelength, SinogramPtr->Obj2Det_Distance, SinogramPtr->Delta_Over_Beta);
	}

	if (TomoInputsPtr->Write2Tiff == 1)
	{
  		dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
  		sprintf(object_file, "%s_n%d", PAG_MAGRET_FILENAME, TomoInputsPtr->node_rank);
  		WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(z_real[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(object_file, "%s_n%d", PAG_PHASERET_FILENAME, TomoInputsPtr->node_rank);
  		WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(z_imag[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
	}    
	size = SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t;
	write_SharedBinFile_At (PAG_MAGRET_FILENAME, &(z_real[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	write_SharedBinFile_At (PAG_PHASERET_FILENAME, &(z_imag[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	
/*	Real_t real_min = z_real[0][0][0], imag_min = z_imag[0][0][0];
	for (i = 0; i < SinogramPtr->N_p; i++)	
	for (j = 0; j < SinogramPtr->N_r; j++)	
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		if (real_min > z_real[i][j][k])
			real_min = z_real[i][j][k];
		if (imag_min > z_imag[i][j][k])
			imag_min = z_imag[i][j][k];
	}	
		
	for (i = 0; i < SinogramPtr->N_p; i++)	
	for (j = 0; j < SinogramPtr->N_r; j++)	
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		z_real[i][j][k] -= real_min;
		z_imag[i][j][k] -= imag_min;
	}

	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Minimum real part = %f imag part = %f of phase retreived images\n", real_min, imag_min);
*/
	for (i = 0; i < SinogramPtr->N_p; i++)	
	for (j = 0; j < SinogramPtr->N_r; j++)	
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		SinogramPtr->MagTomoAux[i][j][k][1] = z_real[i][j][k];	
		SinogramPtr->PhaseTomoAux[i][j][k][1] = z_imag[i][j][k];	
	}
#ifdef ENABLE_TOMO_RECONS
    	initObject(SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
	initErrorSinogam(SinogramPtr, ScannedObjectPtr, TomoInputsPtr);

#ifndef NO_COST_CALCULATE
	    cost = computeCost(SinogramPtr,ScannedObjectPtr,TomoInputsPtr);
	    cost_0_iter = cost;
	    cost_last_iter = cost;
	    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "------------- Iteration 0, Cost = %f------------\n",cost);
#endif
    	for (Iter = 1; Iter <= TomoInputsPtr->NumIter; Iter++)
    	{
      		flag = updateVoxelsTimeSlices (SinogramPtr, ScannedObjectPtr, TomoInputsPtr, Iter, Mask);
#ifndef NO_COST_CALCULATE
	      cost = computeCost(SinogramPtr,ScannedObjectPtr,TomoInputsPtr);
	      percentage_change_in_cost = ((cost - cost_last_iter)/(cost - cost_0_iter))*100.0;
	      check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Percentage change in cost is %f.\n", percentage_change_in_cost);
	      check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "------------- Iteration = %d, Cost = %f ------------\n",Iter,cost);
	      
	      if (cost > cost_last_iter)
		      check_warn(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "Cost value increased.\n");
	      cost_last_iter = cost;
	      /*if (percentage_change_in_cost < TomoInputsPtr->cost_thresh && flag != 0 && Iter > 1){*/
	      if (flag != 0 && Iter > 1){
		        check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Convergence criteria is met.\n");
        		break;
      		}
#else
		if (flag != 0 && Iter > 1){
        		check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Convergence criteria is met.\n");
        		break;
	      }
#endif
	      flag = fflush(TomoInputsPtr->debug_file_ptr);
     		 if (flag != 0)
      			check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cannot flush buffer.\n");
    	}

    	if (TomoInputsPtr->Write2Tiff == 1)
	{
		for (i = 0; i < ScannedObjectPtr->N_time; i++)
	  	{
	    		dimTiff[0] = 1; dimTiff[1] = ScannedObjectPtr->N_z; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;
			sprintf (object_file, "%s_n%d", PAG_MAGOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->MagObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
			sprintf (object_file, "%s_n%d", PAG_PHASEOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->PhaseObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
	  	}
	}

	size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
	write_SharedBinFile_At (PAG_MAGOBJECT_FILENAME, &(ScannedObjectPtr->MagObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	write_SharedBinFile_At (PAG_PHASEOBJECT_FILENAME, &(ScannedObjectPtr->PhaseObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
        sprintf(object_file, "%s_time_%d", MAGOBJECT_FILENAME,0);
	write_SharedBinFile_At (object_file, &(ScannedObjectPtr->MagObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
        sprintf(object_file, "%s_time_%d", PHASEOBJECT_FILENAME,0);
	write_SharedBinFile_At (object_file, &(ScannedObjectPtr->PhaseObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
 #endif 
/*	for (i = 0; i < ScannedObjectPtr->N_time; i++)
  	{
  		free(AMatrixPtr[i].values);
    		free(AMatrixPtr[i].index);
  	}
  
	free (AMatrixPtr);*/
	multifree(ProjLength, 3);	
	multifree(Init, 3);	
	multifree(z_real, 3);
	multifree(z_imag, 3);

	return(0);
}


/*'initErrorSinogram' is used to initialize the error sinogram before start of ICD. It computes e = y - Ax - d. Ax is computed by forward projecting the object x.*/
int32_t initErrorSinogam (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr/*, AMatrixCol* VoxelLineResponse*/)
{
  Real_arr_t*** MagErrorSino = SinogramPtr->MagErrorSino;
  Real_arr_t*** PhaseErrorSino = SinogramPtr->PhaseErrorSino;
  Real_t pixel, magavg = 0, phaseavg = 0;
  int32_t i, j, k, p, sino_idx, slice, flag = 0;
  AMatrixCol* AMatrixPtr = (AMatrixCol*)get_spc(ScannedObjectPtr->N_time, sizeof(AMatrixCol));
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
 /* char error_file[100];*/

  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  {
    AMatrixPtr[i].values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
    AMatrixPtr[i].index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  }
  memset(&(MagErrorSino[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
  memset(&(PhaseErrorSino[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));

#ifdef ENABLE_TOMO_RECONS
  #pragma omp parallel for private(j, k, p, sino_idx, slice, pixel)
  for (i=0; i<ScannedObjectPtr->N_time; i++)
  {
    for (j=0; j<ScannedObjectPtr->N_y; j++)
    {
      for (k=0; k<ScannedObjectPtr->N_x; k++){
        for (p=0; p<ScannedObjectPtr->ProjNum[i]; p++){
          sino_idx = ScannedObjectPtr->ProjIdxPtr[i][p];
          calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, SinogramPtr->DetectorResponse, &(AMatrixPtr[i]), j, k, sino_idx, SinogramPtr->Light_Wavenumber);
          for (slice=0; slice<ScannedObjectPtr->N_z; slice++){
            /*	printf("count = %d, idx = %d, val = %f\n", VoxelLineResponse[slice].count, VoxelLineResponse[slice].index[0], VoxelLineResponse[slice].values[0]);*/
            pixel = (ScannedObjectPtr->PhaseObject[i][slice+1][j][k]); /*slice+1 to account for extra z slices required for MPI*/
            forward_project_voxel (SinogramPtr, pixel, PhaseErrorSino, &(AMatrixPtr[i])/*, &(VoxelLineResponse[slice])*/, sino_idx, slice);
            pixel = (ScannedObjectPtr->MagObject[i][slice+1][j][k]); /*slice+1 to account for extra z slices required for MPI*/
            forward_project_voxel (SinogramPtr, pixel, MagErrorSino, &(AMatrixPtr[i])/*, &(VoxelLineResponse[slice])*/, sino_idx, slice);
          }
        }
      }
    }
  }
#endif

  #pragma omp parallel for private(j, k) reduction(+:magavg,phaseavg)
  for(i = 0; i < SinogramPtr->N_p; i++)
  for (j = 0; j < SinogramPtr->N_r; j++)
  for (k = 0; k < SinogramPtr->N_t; k++)
  {
#ifndef ENABLE_TOMO_RECONS
	MagErrorSino[i][j][k] = SinogramPtr->MagProj[i][j][k];
	PhaseErrorSino[i][j][k] =  SinogramPtr->PhaseProj[i][j][k];
#endif
    	magavg += MagErrorSino[i][j][k];
    	phaseavg += PhaseErrorSino[i][j][k];
  }
  magavg = magavg/(SinogramPtr->N_r*SinogramPtr->N_t*SinogramPtr->N_p);
  phaseavg = phaseavg/(SinogramPtr->N_r*SinogramPtr->N_t*SinogramPtr->N_p);
  check_debug(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "Average of magnitude and phase components of froward projection in node %d are %f and %f\n", TomoInputsPtr->node_rank, magavg, phaseavg);

  #pragma omp parallel for private(j, k)
  for(i = 0; i < SinogramPtr->N_p; i++)
  for (j = 0; j < SinogramPtr->N_r; j++)
  for (k = 0; k < SinogramPtr->N_t; k++)
  {
	SinogramPtr->MagProj[i][j][k] = SinogramPtr->MagTomoAux[i][j][k][1];
	SinogramPtr->PhaseProj[i][j][k] = SinogramPtr->PhaseTomoAux[i][j][k][1];
	if (TomoInputsPtr->recon_type == 2)
	{
		SinogramPtr->MagTomoAux[i][j][k][1] = MagErrorSino[i][j][k];
		SinogramPtr->MagTomoAux[i][j][k][2] = FORWPROJ_ADD_FRACTION*magavg;
		SinogramPtr->MagTomoAux[i][j][k][3] = 2*FORWPROJ_ADD_FRACTION*magavg;
		
		if (j - 1 >= 0)	
			SinogramPtr->MagTomoAux[i][j][k][2] += MagErrorSino[i][j-1][k];
		else
			SinogramPtr->MagTomoAux[i][j][k][2] += MagErrorSino[i][j][k];

		if (j + 1 < SinogramPtr->N_r)
			SinogramPtr->MagTomoAux[i][j][k][3] += MagErrorSino[i][j+1][k];
		else
			SinogramPtr->MagTomoAux[i][j][k][3] += 2*MagErrorSino[i][j][k];

		SinogramPtr->MagTomoAux[i][j][k][0] = (SinogramPtr->MagTomoAux[i][j][k][1] + SinogramPtr->MagTomoAux[i][j][k][2])/2.0;
			
		SinogramPtr->PhaseTomoAux[i][j][k][1] = PhaseErrorSino[i][j][k];	
		SinogramPtr->PhaseTomoAux[i][j][k][2] = FORWPROJ_ADD_FRACTION*phaseavg;	
		SinogramPtr->PhaseTomoAux[i][j][k][3] = 2*FORWPROJ_ADD_FRACTION*phaseavg;	
	
		if (j - 1 >= 0)	
			SinogramPtr->PhaseTomoAux[i][j][k][2] += PhaseErrorSino[i][j-1][k];
		else
			SinogramPtr->PhaseTomoAux[i][j][k][2] += PhaseErrorSino[i][j][k];

		if (j + 1 < SinogramPtr->N_r)
			SinogramPtr->PhaseTomoAux[i][j][k][3] += PhaseErrorSino[i][j+1][k];
		else
			SinogramPtr->PhaseTomoAux[i][j][k][3] += 2*PhaseErrorSino[i][j][k];
	
		SinogramPtr->PhaseTomoAux[i][j][k][0] = (SinogramPtr->PhaseTomoAux[i][j][k][1] + SinogramPtr->PhaseTomoAux[i][j][k][2])/2.0;
    	}

	MagErrorSino[i][j][k] = SinogramPtr->MagTomoAux[i][j][k][1] - SinogramPtr->MagTomoDual[i][j][k] - MagErrorSino[i][j][k];
    	PhaseErrorSino[i][j][k] = SinogramPtr->PhaseTomoAux[i][j][k][1] - SinogramPtr->PhaseTomoDual[i][j][k] - PhaseErrorSino[i][j][k];

	SinogramPtr->MagPRetAux[i][j][k] = exp(-SinogramPtr->MagTomoAux[i][j][k][1])*cos(-SinogramPtr->PhaseTomoAux[i][j][k][1]);
	SinogramPtr->PhasePRetAux[i][j][k] = exp(-SinogramPtr->MagTomoAux[i][j][k][1])*sin(-SinogramPtr->PhaseTomoAux[i][j][k][1]);
	
	SinogramPtr->MagPRetDual[i][j][k] = 0;		
	SinogramPtr->PhasePRetDual[i][j][k] = 0;		
  }
			
  for (i = 0; i < SinogramPtr->N_p; i++)
	compute_phase_projection (SinogramPtr->Measurements_real[i], SinogramPtr->Measurements_imag[i], SinogramPtr->Omega_real[i], SinogramPtr->Omega_imag[i], SinogramPtr->D_real[i], SinogramPtr->D_imag[i], SinogramPtr->MagPRetAux[i], SinogramPtr->PhasePRetAux[i], SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]), SinogramPtr->Light_Wavelength, SinogramPtr->Obj2Det_Distance, SinogramPtr->Freq_Window);
  
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
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Average voxel update over all voxels is %e, total voxels is %e.\n", AverageUpdate, tempTotPix);
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
    char costfile[100] = COST_FILENAME, origcostfile[100] = ORIG_COST_FILENAME, primal_filename[100], dual_filename[100];
    #endif
    Real_t x, y, ar, ai, orig_cost, orig_cost_last, primal_res_real = 0, dual_res_real = 0, primal_res_imag = 0, dual_res_imag = 0, sum_real, sum_imag, real_temp, imag_temp;
    int32_t j, flag = 0, Iter, i, k, HeadIter = 0;
    int dimTiff[4];
    time_t start;
    char detect_file[100] = DETECTOR_RESPONSE_FILENAME;
    char MagUpdateMapFile[100] = UPDATE_MAP_FILENAME, aux_filename[100];
    uint8_t ***Mask;
    Real_arr_t*** omega_abs = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

    Write2Bin (MAG_RECON_PRIMAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &primal_res_real, stdout);
    Write2Bin (MAG_RECON_DUAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &dual_res_real, stdout);
    Write2Bin (PHASE_RECON_PRIMAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &primal_res_imag, stdout);
    Write2Bin (PHASE_RECON_DUAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &dual_res_imag, stdout);
    for (i = 0; i < SinogramPtr->N_p; i++)
    {
	sprintf(primal_filename, "%s_proj_%d", PHASERET_PRIMAL_RESIDUAL_FILENAME, i);
	sprintf(dual_filename, "%s_proj_%d", PHASERET_DUAL_RESIDUAL_FILENAME, i);
    	Write2Bin (primal_filename, 1, 1, 1, 1, sizeof(Real_t), &primal_res_real, stdout);
    	Write2Bin (dual_filename, 1, 1, 1, 1, sizeof(Real_t), &dual_res_real, stdout);
    }

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
    
    DetectorResponseProfile (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
    dimTiff[0] = 1; dimTiff[1] = 1; dimTiff[2] = SinogramPtr->N_p; dimTiff[3] = DETECTOR_RESPONSE_BINS+1;
    sprintf(detect_file, "%s_n%d", detect_file, TomoInputsPtr->node_rank);
    if (TomoInputsPtr->Write2Tiff == 1)
    	if (WriteMultiDimArray2Tiff (detect_file, dimTiff, 0, 1, 2, 3, &(SinogramPtr->DetectorResponse[0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    start = time(NULL);
    
    if (TomoInputsPtr->recon_type == 1)
    {
/*	    gen_data_GroundTruth (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);*/
	    do_PagPhaseRet_MBIRRecon (SinogramPtr, ScannedObjectPtr, TomoInputsPtr, Mask);
    }
    else if (TomoInputsPtr->recon_type == 2)
    {

    if (initObject(SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;
    if (initErrorSinogam(SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;
/*    if (init_minmax_object (ScannedObjectPtr, TomoInputsPtr)) goto error;*/

    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Time taken to initialize object and compute error sinogram = %fmins\n", difftime(time(NULL),start)/60.0);
  
    start=time(NULL);
#ifdef ENABLE_TOMO_RECONS
    orig_cost_last = compute_original_cost(SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
    check_info(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "HeadIter = 0: The original cost value is %f. \n", orig_cost_last);
    if (TomoInputsPtr->node_rank == 0)
	   Write2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost_last, TomoInputsPtr->debug_file_ptr);
#endif

    for (HeadIter = 1; HeadIter <= TomoInputsPtr->Head_MaxIter; HeadIter++)
    {
#ifdef ENABLE_TOMO_RECONS
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
#endif
/*    	if (initErrorSinogam(SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;*/
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

		estimate_complex_projection (SinogramPtr->Measurements_real[i], SinogramPtr->Measurements_imag[i], SinogramPtr->Omega_real[i], SinogramPtr->Omega_imag[i], SinogramPtr->D_real[i], SinogramPtr->D_imag[i], SinogramPtr->MagTomoAux[i], SinogramPtr->PhaseTomoAux[i], TomoInputsPtr->Weight[i], SinogramPtr->MagErrorSino[i], SinogramPtr->PhaseErrorSino[i], SinogramPtr->MagPRetAux[i], SinogramPtr->PhasePRetAux[i], SinogramPtr->MagPRetDual[i], SinogramPtr->PhasePRetDual[i], SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, TomoInputsPtr->NMS_rho, TomoInputsPtr->NMS_chi, TomoInputsPtr->NMS_gamma, TomoInputsPtr->NMS_sigma, TomoInputsPtr->NMS_threshold, TomoInputsPtr->NMS_MaxIter, TomoInputsPtr->SteepDes_threshold, TomoInputsPtr->SteepDes_MaxIter, TomoInputsPtr->PRet_threshold, TomoInputsPtr->PRet_MaxIter, TomoInputsPtr->ADMM_mu, TomoInputsPtr->ADMM_nu, SinogramPtr->fftforw_arr[i], &(SinogramPtr->fftforw_plan[i]), SinogramPtr->fftback_arr[i], &(SinogramPtr->fftback_plan[i]), SinogramPtr->Light_Wavelength, SinogramPtr->Obj2Det_Distance, SinogramPtr->Freq_Window, i, ScannedObjectPtr, TomoInputsPtr);
	
		for (j = 0; j < SinogramPtr->N_r; j++)
		for (k = 0; k < SinogramPtr->N_t; k++)
		{
			SinogramPtr->MagErrorSino[i][j][k] = SinogramPtr->MagTomoAux[i][j][k][1] - SinogramPtr->MagErrorSino[i][j][k];
			SinogramPtr->PhaseErrorSino[i][j][k] = SinogramPtr->PhaseTomoAux[i][j][k][1] - SinogramPtr->PhaseErrorSino[i][j][k];
		}
	}

#ifdef ENABLE_TOMO_RECONS
	primal_res_real = 0; dual_res_real = 0; primal_res_imag = 0; dual_res_imag = 0; sum_real = 0, sum_imag = 0;
        #pragma omp parallel for collapse(3) private(i,j,k,ar,ai,real_temp,imag_temp) reduction(+:primal_res_real,dual_res_real,primal_res_imag,dual_res_imag,sum_real,sum_imag)
	for (i = 0; i < SinogramPtr->N_p; i++)
	for (j = 0; j < SinogramPtr->N_r; j++)
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		ar = SinogramPtr->MagTomoDual[i][j][k]; ai = SinogramPtr->PhaseTomoDual[i][j][k];
		SinogramPtr->MagTomoDual[i][j][k] = -SinogramPtr->MagErrorSino[i][j][k];
		SinogramPtr->PhaseTomoDual[i][j][k] = -SinogramPtr->PhaseErrorSino[i][j][k];
		SinogramPtr->MagErrorSino[i][j][k] += ar - SinogramPtr->MagTomoDual[i][j][k];
		SinogramPtr->PhaseErrorSino[i][j][k] += ai - SinogramPtr->PhaseTomoDual[i][j][k];
		
		real_temp = SinogramPtr->MagTomoDual[i][j][k] + SinogramPtr->MagErrorSino[i][j][k];
		imag_temp = SinogramPtr->PhaseTomoDual[i][j][k] + SinogramPtr->PhaseErrorSino[i][j][k];
		primal_res_real += real_temp*real_temp;
		primal_res_imag += imag_temp*imag_temp;
		real_temp = SinogramPtr->MagTomoAux[i][j][k][1] - SinogramPtr->MagProj[i][j][k];
		imag_temp = SinogramPtr->PhaseTomoAux[i][j][k][1] - SinogramPtr->PhaseProj[i][j][k];
		dual_res_real += real_temp*real_temp;
		dual_res_imag += imag_temp*imag_temp;
		sum_real += SinogramPtr->MagTomoDual[i][j][k]*SinogramPtr->MagTomoDual[i][j][k];
		sum_imag += SinogramPtr->PhaseTomoDual[i][j][k]*SinogramPtr->PhaseTomoDual[i][j][k];

		SinogramPtr->MagProj[i][j][k] = SinogramPtr->MagTomoAux[i][j][k][1];
		SinogramPtr->PhaseProj[i][j][k] = SinogramPtr->PhaseTomoAux[i][j][k][1];
    	}
	dual_res_real = sqrt(dual_res_real/sum_real);
	dual_res_imag = sqrt(dual_res_imag/sum_imag);
	primal_res_real = sqrt(primal_res_real/(SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t));
	primal_res_imag = sqrt(primal_res_imag/(SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t));
	
	Append2Bin (MAG_RECON_PRIMAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &primal_res_real, stdout);
	Append2Bin (MAG_RECON_DUAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &dual_res_real, stdout);
	Append2Bin (PHASE_RECON_PRIMAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &primal_res_imag, stdout);
	Append2Bin (PHASE_RECON_DUAL_RESIDUAL_FILENAME, 1, 1, 1, 1, sizeof(Real_t), &dual_res_imag, stdout);

	orig_cost = compute_original_cost(SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
        check_info(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "HeadIter = %d: The original cost value is %f. The decrease in original cost is %f.\n", HeadIter, orig_cost, orig_cost_last - orig_cost);
    	if (TomoInputsPtr->node_rank == 0)
	   Append2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost, TomoInputsPtr->debug_file_ptr);
	
	if (orig_cost > orig_cost_last)
      		check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cost of original cost function increased!\n");
	orig_cost_last = orig_cost;
	
    	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Head Iter = %d, mag primal residual = %e, mag dual residual = %e, phase primal residual = %e, phase dual residual = %e\n", HeadIter, primal_res_real, dual_res_real, primal_res_imag, dual_res_imag);
#endif
  	
	dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
  	if (TomoInputsPtr->Write2Tiff == 1)
  	{
  		sprintf(aux_filename, "%s_n%d", MAGTOMOAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->MagTomoAux[0][0][0][0]), 0, 1, 4, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASETOMOAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->PhaseTomoAux[0][0][0][0]), 0, 1, 4, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", MAGTOMODUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->MagTomoDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASETOMODUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->PhaseTomoDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		
		sprintf(aux_filename, "%s_n%d", MAGPRETAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->MagPRetAux[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASEPRETAUX_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->PhasePRetAux[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", MAGPRETDUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag = WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->MagPRetDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  		sprintf(aux_filename, "%s_n%d", PHASEPRETDUAL_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(SinogramPtr->PhasePRetDual[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);

		for (i = 0; i < SinogramPtr->N_p; i++)
		for (j = 0; j < SinogramPtr->N_r; j++)
		for (k = 0; k < SinogramPtr->N_t; k++)
			omega_abs[i][j][k] = sqrt(SinogramPtr->Omega_real[i][j][k]*SinogramPtr->Omega_real[i][j][k] + SinogramPtr->Omega_imag[i][j][k]*SinogramPtr->Omega_imag[i][j][k]);
  		
		sprintf(aux_filename, "%s_n%d", OMEGAABS_FILENAME, TomoInputsPtr->node_rank);
  		flag |= WriteMultiDimArray2Tiff (aux_filename, dimTiff, 0, 1, 2, 3, &(omega_abs[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
  	}

/*	if (avg_head_update < TomoInputsPtr->Head_threshold && HeadIter > 1)
		break;*/
    }  

    }

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
	if (write_SharedBinFile_At (MAGTOMODUAL_FILENAME, &(SinogramPtr->MagTomoDual[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error; 
	if (write_SharedBinFile_At (PHASETOMODUAL_FILENAME, &(SinogramPtr->PhaseTomoDual[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
	if (write_SharedBinFile_At (MAGPRETDUAL_FILENAME, &(SinogramPtr->MagPRetDual[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error; 
	if (write_SharedBinFile_At (PHASEPRETDUAL_FILENAME, &(SinogramPtr->PhasePRetDual[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
 
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

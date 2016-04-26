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
int updateVoxelsTimeSlices(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, int32_t Iter, uint8_t** Mask);

/*computes the location of (i,j,k) th element in a 1D array*/
int32_t array_loc_1D (int32_t i, int32_t j, int32_t k, int32_t N_j, int32_t N_k)
{
  return (i*N_j*N_k + j*N_k + k);
}

/*computes the value of cost function. 'ErrorSino' is the error sinogram*/
Real_t computeCost(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  Real_t cost=0, temp=0, forward=0, prior=0;
  Real_t Diff;
  int32_t i,j,k,p,N_z,cidx;
  bool j_minus, k_minus, j_plus, k_plus, p_plus;
 
  #pragma omp parallel for private(j, k, temp) reduction(+:cost)
  for (i = 0; i < SinogramPtr->N_p; i++)
  for (j = 0; j < SinogramPtr->N_r; j++)
  for (k = 0; k < SinogramPtr->N_t; k++)
  {
    temp = SinogramPtr->ErrorSino_Unflip_x[i][j][k];
    cost += temp*temp*TomoInputsPtr->Weight;
    temp = SinogramPtr->ErrorSino_Flip_x[i][j][k];
    cost += temp*temp*TomoInputsPtr->Weight;
    temp = SinogramPtr->ErrorSino_Unflip_y[i][j][k];
    cost += temp*temp*TomoInputsPtr->Weight;
    temp = SinogramPtr->ErrorSino_Flip_y[i][j][k];
    cost += temp*temp*TomoInputsPtr->Weight;
  }
  cost /= 2.0;
  /*When computing the cost of the prior term it is important to make sure that you don't include the cost of any pair of neighbors more than once. In this code, a certain sense of causality is used to compute the cost. We also assume that the weghting kernel given by 'Filter' is symmetric. Let i, j and k correspond to the three dimensions. If we go forward to i+1, then all neighbors at j-1, j, j+1, k+1, k, k-1 are to be considered. However, if for the same i, if we go forward to j+1, then all k-1, k, and k+1 should be considered. For same i and j, only the neighbor at k+1 is considred.*/
  temp = 0;
  N_z = ScannedObjectPtr->N_z + 2;
  if (TomoInputsPtr->node_rank == TomoInputsPtr->node_num-1)
  N_z = ScannedObjectPtr->N_z + 1;
  #pragma omp parallel for private(Diff, p, j, k, j_minus, k_minus, p_plus, j_plus, k_plus, cidx) reduction(+:temp)
  for (p = 1; p < ScannedObjectPtr->N_z + 1; p++)
  for (j = 0; j < ScannedObjectPtr->N_y; j++)
  {
    for (k = 0; k < ScannedObjectPtr->N_x; k++)
    {
      j_minus = (j - 1 >= 0)? true : false;
      k_minus = (k - 1 >= 0)? true : false;
      
      p_plus = (p + 1 < N_z)? true : false;
      j_plus = (j + 1 < ScannedObjectPtr->N_y)? true : false;
      k_plus = (k + 1 < ScannedObjectPtr->N_x)? true : false;
      
      if(k_plus == true) {
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = (ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p][j][k + 1][cidx]);
        	temp += TomoInputsPtr->Spatial_Filter[1][1][2] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	}
        Diff = (ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p][j][k + 1]);
        temp += TomoInputsPtr->Spatial_Filter[1][1][2] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
      }
      if(j_plus == true) {
        if(k_minus == true) {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = (ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p][j + 1][k - 1][cidx]);
          	temp += TomoInputsPtr->Spatial_Filter[1][2][0] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
          }
	  Diff = (ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p][j + 1][k - 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][0] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
        }
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = (ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p][j + 1][k][cidx]);
        	temp += TomoInputsPtr->Spatial_Filter[1][2][1] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	}
        Diff = (ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p][j + 1][k]);
        temp += TomoInputsPtr->Spatial_Filter[1][2][1] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);

        if(k_plus == true) {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = (ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p][j + 1][k + 1][cidx]);
          	temp += TomoInputsPtr->Spatial_Filter[1][2][2] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
          }
	  Diff = (ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p][j + 1][k + 1]);
          temp += TomoInputsPtr->Spatial_Filter[1][2][2] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
        }
      }
      if (p_plus == true)
      {
        if(j_minus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p + 1][j - 1][k][cidx];
          	temp += TomoInputsPtr->Spatial_Filter[2][0][1] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	  }
          Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p + 1][j - 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][0][1] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
        }
        
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p+1][j][k][cidx];
        	temp += TomoInputsPtr->Spatial_Filter[2][1][1] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	}
        Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p+1][j][k];
        temp += TomoInputsPtr->Spatial_Filter[2][1][1] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
        if(j_plus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p+1][j + 1][k][cidx];
          	temp += TomoInputsPtr->Spatial_Filter[2][2][1] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	  }
          Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p+1][j + 1][k];
          temp += TomoInputsPtr->Spatial_Filter[2][2][1] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
        }
        if(j_minus == true)
        {
          if(k_minus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p + 1][j - 1][k - 1][cidx];
            	temp += TomoInputsPtr->Spatial_Filter[2][0][0] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	    }
            Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p + 1][j - 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][0] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
          }
          if(k_plus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p + 1][j - 1][k + 1][cidx];
            	temp += TomoInputsPtr->Spatial_Filter[2][0][2] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	    }
            Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p + 1][j - 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][0][2] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
          }
        }
        if(k_minus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p + 1][j][k - 1][cidx];
          	temp += TomoInputsPtr->Spatial_Filter[2][1][0] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	  }
          Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p + 1][j][k - 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][0] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
        }
        if(j_plus == true)
        {
          if(k_minus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p + 1][j + 1][k - 1][cidx];
            	temp += TomoInputsPtr->Spatial_Filter[2][2][0] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	    }
            Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p + 1][j + 1][k - 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][0] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
          }
          if(k_plus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p + 1][j + 1][k + 1][cidx];
            	temp += TomoInputsPtr->Spatial_Filter[2][2][2] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	    }
            Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p + 1][j + 1][k + 1];
            temp += TomoInputsPtr->Spatial_Filter[2][2][2] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
          }
        }
        if(k_plus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ScannedObjectPtr->MagPotentials[p][j][k][cidx] - ScannedObjectPtr->MagPotentials[p + 1][j][k + 1][cidx];
          	temp += TomoInputsPtr->Spatial_Filter[2][1][2] * QGGMRF_Value(Diff,TomoInputsPtr->Mag_Sigma_Q[cidx], TomoInputsPtr->Mag_Sigma_Q_P[cidx], ScannedObjectPtr->Mag_C[cidx]);
	  }
          Diff = ScannedObjectPtr->ElecPotentials[p][j][k] - ScannedObjectPtr->ElecPotentials[p + 1][j][k + 1];
          temp += TomoInputsPtr->Spatial_Filter[2][1][2] * QGGMRF_Value(Diff,TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C);
        }
      }
    }
  }
  /*Use MPI reduction operation to add the forward and prior costs from all nodes*/
  MPI_Reduce(&cost, &forward, 1, MPI_REAL_DATATYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&temp, &prior, 1, MPI_REAL_DATATYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (TomoInputsPtr->node_rank == 0)
  {
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Error sino cost = %f\n",forward);
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Decrease in error sino cost = %f\n",TomoInputsPtr->ErrorSino_Cost-forward);
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


/*randomly select the voxels lines which need to be updated along the x-y plane for each z-block and time slice*/
void randomly_select_x_y (ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, uint8_t** Mask)
{
  int32_t j, num,n, Index, col, row, *Counter, ArraySize, block;
  ArraySize = ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
  Counter = (int32_t*)get_spc(ArraySize, sizeof(int32_t));
  for (block=0; block<TomoInputsPtr->num_z_blocks; block++)
  {
    ArraySize = ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
    for (Index = 0; Index < ArraySize; Index++)
    Counter[Index] = Index;
    
    TomoInputsPtr->UpdateSelectNum[block] = 0;
    for (j=0; j<ScannedObjectPtr->N_x*ScannedObjectPtr->N_y; j++){
      Index = floor(random2() * ArraySize);
      Index = (Index == ArraySize)?ArraySize-1:Index;
      col = Counter[Index] % ScannedObjectPtr->N_x;
      row = Counter[Index] / ScannedObjectPtr->N_x;
      for (n = block*(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks); n < (block+1)*(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks); n++)
      if (Mask[row][col] == 1)
      {
        num = TomoInputsPtr->UpdateSelectNum[block];
        TomoInputsPtr->x_rand_select[block][num] = col;
        TomoInputsPtr->y_rand_select[block][num] = row;
        (TomoInputsPtr->UpdateSelectNum[block])++;
        break;
      }
      Counter[Index] = Counter[ArraySize - 1];
      ArraySize--;
    }
  }
  free(Counter);
}



/*'initErrorSinogram' is used to initialize the error sinogram before start of ICD. It computes e = y - Ax - d. Ax is computed by forward projecting the object x.*/
int32_t initErrorSinogam (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr/*, AMatrixCol* VoxelLineResponse*/)
{
  Real_arr_t*** ErrorSino_Unflip_x = SinogramPtr->ErrorSino_Unflip_x;
  Real_arr_t*** ErrorSino_Flip_x = SinogramPtr->ErrorSino_Flip_x;
  Real_arr_t*** ErrorSino_Unflip_y = SinogramPtr->ErrorSino_Unflip_y;
  Real_arr_t*** ErrorSino_Flip_y = SinogramPtr->ErrorSino_Flip_y;
  
  Real_t unflipavg = 0, flipavg = 0;
  int32_t i, j, k, sino_idx, slice, flag = 0;
  AMatrixCol AMatrixPtr_X, AMatrixPtr_Y;
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
 /* char error_file[100];*/

  AMatrixPtr_X.values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
  AMatrixPtr_X.index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  AMatrixPtr_Y.values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
  AMatrixPtr_Y.index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  
  memset(&(ErrorSino_Unflip_x[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
  memset(&(ErrorSino_Flip_x[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
  memset(&(ErrorSino_Unflip_y[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));
  memset(&(ErrorSino_Flip_y[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));

/*  #pragma omp parallel for private(j, k, sino_idx, slice)*/
    for (j=0; j<ScannedObjectPtr->N_y; j++)
    {
      for (k=0; k<ScannedObjectPtr->N_x; k++){
        for (sino_idx=0; sino_idx < SinogramPtr->N_p; sino_idx++){
          calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, SinogramPtr->DetectorResponse, &(AMatrixPtr_X), j, k, sino_idx);
          for (slice=0; slice<ScannedObjectPtr->N_z; slice++){
          	calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, SinogramPtr->DetectorResponse, &(AMatrixPtr_Y), j, slice, sino_idx);
            /*	printf("count = %d, idx = %d, val = %f\n", VoxelLineResponse[slice].count, VoxelLineResponse[slice].index[0], VoxelLineResponse[slice].values[0]);*/
		forward_project_voxel (SinogramPtr, ScannedObjectPtr->MagPotentials[slice+1][j][k], ScannedObjectPtr->ElecPotentials[slice+1][j][k], ErrorSino_Unflip_x, ErrorSino_Flip_x, &(AMatrixPtr_X), sino_idx, slice);
		forward_project_voxel (SinogramPtr, ScannedObjectPtr->MagPotentials[slice+1][j][k], ScannedObjectPtr->ElecPotentials[slice+1][j][k], ErrorSino_Unflip_y, ErrorSino_Flip_y, &(AMatrixPtr_Y), sino_idx, k);
          }
        }
      }
    }
  

  #pragma omp parallel for private(j, k) reduction(+:unflipavg,flipavg)
  for(i = 0; i < SinogramPtr->N_p; i++)
  for(j = 0; j < SinogramPtr->N_r; j++)
  for(k = 0; k < SinogramPtr->N_t; k++)
  {
    	unflipavg += ErrorSino_Unflip_x[i][j][k];
    	flipavg += ErrorSino_Flip_x[i][j][k];
    	unflipavg += ErrorSino_Unflip_y[i][j][k];
    	flipavg += ErrorSino_Flip_y[i][j][k];
  }
  unflipavg = unflipavg/(SinogramPtr->N_r*SinogramPtr->N_t*SinogramPtr->N_p);
  flipavg = flipavg/(SinogramPtr->N_r*SinogramPtr->N_t*SinogramPtr->N_p);
  check_debug(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "Average of unflipped and flipped components of forward projection in node %d are %f and %f\n", TomoInputsPtr->node_rank, unflipavg, flipavg);

  #pragma omp parallel for private(j, k)
  for(i = 0; i < SinogramPtr->N_p; i++)
  for (j = 0; j < SinogramPtr->N_r; j++)
  for (k = 0; k < SinogramPtr->N_t; k++)
  {
	ErrorSino_Unflip_x[i][j][k] = SinogramPtr->Data_Unflip_x[i][j][k] - ErrorSino_Unflip_x[i][j][k];
	ErrorSino_Flip_x[i][j][k] = SinogramPtr->Data_Flip_x[i][j][k] - ErrorSino_Flip_x[i][j][k];
	ErrorSino_Unflip_y[i][j][k] = SinogramPtr->Data_Unflip_y[i][j][k] - ErrorSino_Unflip_y[i][j][k];
	ErrorSino_Flip_y[i][j][k] = SinogramPtr->Data_Flip_y[i][j][k] - ErrorSino_Flip_y[i][j][k];
  }
			
  free(AMatrixPtr_X.values);
  free(AMatrixPtr_X.index);
  free(AMatrixPtr_Y.values);
  free(AMatrixPtr_Y.index);
  return (flag);
}


  /*Implements mutithreaded shared memory parallelization using OpenMP and splits work among
  threads. Each thread gets a certain time slice and z block to update.
  Multithreading is done within the z-blocks assigned to each node.
  ErrorSino - Error sinogram
  Iter - Present iteration number
  MagUpdateMap - Magnitude update map containing the magnitude of update of each voxel
  Mask - If a certain element is true then the corresponding voxel is updated*/
  int updateVoxelsTimeSlices(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, int32_t Iter, uint8_t** Mask)
  {
    Real_t AverageUpdate = 0, tempUpdate, avg_update_percentage, total_vox_mag = 0.0, vox_mag = 0.0;
    int32_t xy_start, xy_end, j, K, block, idx, *z_start, *z_stop;
    Real_t tempTotPix = 0, total_pix = 0;
    long int *zero_count, total_zero_count = 0;
    int32_t* thread_num = (int32_t*)get_spc(TomoInputsPtr->num_z_blocks, sizeof(int32_t));
    MPI_Request mag_send_reqs, mag_recv_reqs, elec_send_reqs, elec_recv_reqs;
    
    z_start = (int32_t*)get_spc(TomoInputsPtr->num_z_blocks, sizeof(int32_t));
    z_stop = (int32_t*)get_spc(TomoInputsPtr->num_z_blocks, sizeof(int32_t));
    
    randomly_select_x_y (ScannedObjectPtr, TomoInputsPtr, Mask);
    
    zero_count = (long int*)get_spc(TomoInputsPtr->num_z_blocks, sizeof(long int));
    
    memset(&(zero_count[0]), 0, TomoInputsPtr->num_z_blocks*sizeof(long int));
    /*	K = ScannedObjectPtr->N_time*ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
    K = (K - total_zero_count)/(ScannedObjectPtr->gamma*K);*/
    K = ScannedObjectPtr->NHICD_Iterations;
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of NHICD iterations is %d.\n", K);
    for (j = 0; j < K; j++)
    {
      total_vox_mag = 0.0;
      
      /*#pragma omp parallel for private(block, idx, xy_start, xy_end) reduction(+:total_vox_mag)*/
      for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
      {
        idx = block;
        z_start[idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
        z_stop[idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
        z_stop[idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[idx];
        xy_start = j*floor(TomoInputsPtr->UpdateSelectNum[idx]/K);
        xy_end = (j + 1)*floor(TomoInputsPtr->UpdateSelectNum[idx]/K) - 1;
        xy_end = (j == K - 1) ? TomoInputsPtr->UpdateSelectNum[idx] - 1: xy_end;
        /*	printf ("Loop 1 Start - j = %d, i = %d, idx = %d, z_start = %d, z_stop = %d, xy_start = %d, xy_end = %d\n", j, i, idx, z_start[i][idx], z_stop[i][idx], xy_start, xy_end);*/
        total_vox_mag += updateVoxels (z_start[idx], z_stop[idx], xy_start, xy_end, TomoInputsPtr->x_rand_select[idx], TomoInputsPtr->y_rand_select[idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->ErrorSino_Unflip_x, SinogramPtr->ErrorSino_Flip_x, SinogramPtr->ErrorSino_Unflip_y, SinogramPtr->ErrorSino_Flip_y, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[idx]), ScannedObjectPtr->UpdateMap[idx], Mask);
        thread_num[idx] = omp_get_thread_num();
      }
      
      /*check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Send MPI info\n");*/
      MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 0);
      /*	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "update_Sinogram_Offset: Will compute projection offset error\n");*/
      MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 0);
 /*     #pragma omp parallel for private(block, idx, xy_start, xy_end) reduction(+:total_vox_mag)*/
      for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
      {
        idx = block + 1;
        z_start[idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
        z_stop[idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
        z_stop[idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[idx];
        xy_start = j*floor(TomoInputsPtr->UpdateSelectNum[idx]/K);
        xy_end = (j + 1)*floor(TomoInputsPtr->UpdateSelectNum[idx]/K) - 1;
        xy_end = (j == K - 1) ? TomoInputsPtr->UpdateSelectNum[idx] - 1: xy_end;
        total_vox_mag += updateVoxels (z_start[idx], z_stop[idx], xy_start, xy_end, TomoInputsPtr->x_rand_select[idx], TomoInputsPtr->y_rand_select[idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->ErrorSino_Unflip_x, SinogramPtr->ErrorSino_Flip_x, SinogramPtr->ErrorSino_Unflip_y, SinogramPtr->ErrorSino_Flip_y, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[idx]), ScannedObjectPtr->UpdateMap[idx], Mask);
        thread_num[idx] = omp_get_thread_num();
        /*	printf ("Loop 2 - i = %d, idx = %d, z_start = %d, z_stop = %d, xy_start = %d, xy_end = %d\n", i, idx, z_start[i][idx], z_stop[i][idx], xy_start, xy_end);*/
      }
      
      MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 1);
      MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 1);
      VSC_based_Voxel_Line_Select(ScannedObjectPtr, TomoInputsPtr, ScannedObjectPtr->UpdateMap);
      /*	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of NHICD voxel lines to be updated in iteration %d is %d\n", j, num_voxel_lines);*/
      if (Iter > 1 && TomoInputsPtr->no_NHICD == 0)
      {
 /*       #pragma omp parallel for private(block, idx)*/
        for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
        {
          idx = block;
          z_start[idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
          z_stop[idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
          z_stop[idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[idx];
          updateVoxels (z_start[idx], z_stop[idx], 0, TomoInputsPtr->NHICDSelectNum[idx]-1, TomoInputsPtr->x_NHICD_select[idx], TomoInputsPtr->y_NHICD_select[idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->ErrorSino_Unflip_x, SinogramPtr->ErrorSino_Flip_x, SinogramPtr->ErrorSino_Unflip_y, SinogramPtr->ErrorSino_Flip_y, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[idx]), ScannedObjectPtr->UpdateMap[idx], Mask);
          thread_num[idx] = omp_get_thread_num();
          /*	printf ("Loop 1 NHICD - i = %d, idx = %d, z_start = %d, z_stop = %d\n", i, idx, z_start[i][idx], z_stop[i][idx]);*/
        }
        
        MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 0);
        MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 0);
        
/*        #pragma omp parallel for private(block, idx)*/
        for (block = 0; block < TomoInputsPtr->num_z_blocks; block = block + 2)
        {
          idx = block + 1;
          z_start[idx] = idx*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
          z_stop[idx] = (idx + 1)*floor(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks) - 1;
          z_stop[idx] = (idx >= TomoInputsPtr->num_z_blocks - 1) ? ScannedObjectPtr->N_z - 1: z_stop[idx];
          updateVoxels (z_start[idx], z_stop[idx], 0, TomoInputsPtr->NHICDSelectNum[idx]-1, TomoInputsPtr->x_NHICD_select[idx], TomoInputsPtr->y_NHICD_select[idx], SinogramPtr, ScannedObjectPtr, TomoInputsPtr, SinogramPtr->ErrorSino_Unflip_x, SinogramPtr->ErrorSino_Flip_x, SinogramPtr->ErrorSino_Unflip_y, SinogramPtr->ErrorSino_Flip_y, SinogramPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, &(zero_count[idx]), ScannedObjectPtr->UpdateMap[idx], Mask);
          thread_num[idx] = omp_get_thread_num();
          /*	printf ("Loop 2 NHICD - i = %d, idx = %d, z_start = %d, z_stop = %d\n", i, idx, z_start[i][idx], z_stop[i][idx]);*/
        }
        
        MPI_Send_Recv_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 1);
        MPI_Wait_Z_Slices (ScannedObjectPtr, TomoInputsPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 1);
      }
    }
    
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Time Slice, Z Start, Z End - Thread : ");
    total_pix = 0;
      for (block=0; block<TomoInputsPtr->num_z_blocks; block++){
        total_pix += TomoInputsPtr->UpdateSelectNum[block]*(ScannedObjectPtr->N_z/TomoInputsPtr->num_z_blocks);
        for (j=0; j<TomoInputsPtr->UpdateSelectNum[block]; j++){
          AverageUpdate += ScannedObjectPtr->UpdateMap[block][TomoInputsPtr->y_rand_select[block][j]][TomoInputsPtr->x_rand_select[block][j]];
        }
        total_zero_count += zero_count[block];
        check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "%d,%d-%d; ", z_start[block], z_stop[block], thread_num[block]);
      }
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "\n");
    
    MPI_Allreduce(&AverageUpdate, &tempUpdate, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_pix, &tempTotPix, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_vox_mag, &vox_mag, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    AverageUpdate = tempUpdate/(tempTotPix);
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Average voxel update over all voxels is %e, total voxels is %e.\n", AverageUpdate, tempTotPix);
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Zero count is %ld.\n", total_zero_count);
    
    free(zero_count);
    free(thread_num);
    free(z_start);
    free(z_stop);
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
    char costfile[100] = COST_FILENAME;
    #endif
    Real_t x, y;
    int32_t j, flag = 0, Iter, k;
    int dimTiff[4];
    time_t start;
    char detect_file[100] = DETECTOR_RESPONSE_FILENAME;
    char MagUpdateMapFile[100] = UPDATE_MAP_FILENAME;
    uint8_t **Mask;

    /*AMatrixCol *VoxelLineResponse;*/
    #ifdef POSITIVITY_CONSTRAINT
    	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Enforcing positivity constraint\n");
    #endif
    
    ScannedObjectPtr->UpdateMap = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
    SinogramPtr->DetectorResponse = (Real_arr_t **)multialloc(sizeof(Real_arr_t), 2, SinogramPtr->N_p, DETECTOR_RESPONSE_BINS + 1);

    SinogramPtr->ErrorSino_Unflip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
    SinogramPtr->ErrorSino_Flip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

    SinogramPtr->ErrorSino_Unflip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
    SinogramPtr->ErrorSino_Flip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

    Mask = (uint8_t**)multialloc(sizeof(uint8_t), 2, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
    
    memset(&(ScannedObjectPtr->UpdateMap[0][0][0]), 0, TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x*sizeof(Real_arr_t));
/*    omp_set_num_threads(TomoInputsPtr->num_threads);*/
    check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of CPU cores is %d\n", (int)omp_get_num_procs());
    /*	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ICD_BackProject: Number of threads is %d\n", TomoInputsPtr->num_threads) ;*/
    for (j = 0; j < ScannedObjectPtr->N_y; j++)
    for (k = 0; k < ScannedObjectPtr->N_x; k++){
      x = ScannedObjectPtr->x0 + ((Real_t)k + 0.5)*ScannedObjectPtr->delta_xy;
      y = ScannedObjectPtr->y0 + ((Real_t)j + 0.5)*ScannedObjectPtr->delta_xy;
      if (x*x + y*y < TomoInputsPtr->radius_obj*TomoInputsPtr->radius_obj)
        Mask[j][k] = 1;
      else
        Mask[j][k] = 0;
    }
    
    DetectorResponseProfile (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
    dimTiff[0] = 1; dimTiff[1] = 1; dimTiff[2] = SinogramPtr->N_p; dimTiff[3] = DETECTOR_RESPONSE_BINS+1;
    sprintf(detect_file, "%s_n%d", detect_file, TomoInputsPtr->node_rank);
    if (TomoInputsPtr->Write2Tiff == 1)
    	if (WriteMultiDimArray2Tiff (detect_file, dimTiff, 0, 1, 2, 3, &(SinogramPtr->DetectorResponse[0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    start = time(NULL);
    
    if (initObject(SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;
    if (initErrorSinogam(SinogramPtr, ScannedObjectPtr, TomoInputsPtr)) goto error;
/*    if (init_minmax_object (ScannedObjectPtr, TomoInputsPtr)) goto error;*/

    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Time taken to initialize object and compute error sinogram = %fmins\n", difftime(time(NULL),start)/60.0);
  
    start=time(NULL);
/*    if (TomoInputsPtr->node_rank == 0)
	   Write2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost_last, TomoInputsPtr->debug_file_ptr);

    for (HeadIter = 1; HeadIter <= TomoInputsPtr->Head_MaxIter; HeadIter++)
    {*/
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

     /*   check_info(TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "HeadIter = %d: The original cost value is %f. The decrease in original cost is %f.\n", HeadIter, orig_cost, orig_cost_last - orig_cost);
    	if (TomoInputsPtr->node_rank == 0)
	   Append2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost, TomoInputsPtr->debug_file_ptr);
	
	if (orig_cost > orig_cost_last)
      		check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cost of original cost function increased!\n");
	orig_cost_last = orig_cost;
	
	if (avg_head_update < TomoInputsPtr->Head_threshold && HeadIter > 1)
		break;
    }*/

    int32_t size = TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
    if (write_SharedBinFile_At (MagUpdateMapFile, &(ScannedObjectPtr->UpdateMap[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) goto error;
    
    dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
    if (TomoInputsPtr->Write2Tiff == 1)
    {
    	if (WriteMultiDimArray2Tiff (ERRORSINO_UNFLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->ErrorSino_Unflip_x[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    	if (WriteMultiDimArray2Tiff (ERRORSINO_FLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->ErrorSino_Flip_x[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    	if (WriteMultiDimArray2Tiff (ERRORSINO_UNFLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->ErrorSino_Unflip_y[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    	if (WriteMultiDimArray2Tiff (ERRORSINO_FLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->ErrorSino_Flip_y[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    }

    multifree(SinogramPtr->ErrorSino_Unflip_x, 3);
    multifree(SinogramPtr->ErrorSino_Flip_x, 3);
    multifree(SinogramPtr->ErrorSino_Unflip_y, 3);
    multifree(SinogramPtr->ErrorSino_Flip_y, 3);

    multifree(SinogramPtr->DetectorResponse, 2);
    multifree(Mask, 2);
    multifree(ScannedObjectPtr->UpdateMap, 3);
 
    check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Finished running ICD_BackProject.\n");
    flag = fflush(TomoInputsPtr->debug_file_ptr);
    if (flag != 0)
       check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cannot flush buffer.\n");
    
    return(0);

error:
    multifree(SinogramPtr->ErrorSino_Unflip_x, 3);
    multifree(SinogramPtr->ErrorSino_Flip_x, 3);
    multifree(SinogramPtr->DetectorResponse,2);
    multifree(Mask,2);
    multifree(ScannedObjectPtr->UpdateMap, 3);
    return(-1);
  }

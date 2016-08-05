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
/*#include "XT_NHICD.h"*/
#include "omp.h"
/*#include "XT_MPI.h"*/
/*#include <mpi.h>*/
#include "XT_VoxUpdate.h"
#include "XT_ForwardProject.h"
/*#include "XT_MPIIO.h"*/
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
#include "XT_MagElecDen.h" 
#include "XT_DensityUpdate.h"

int32_t initErrorSinogam(Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr);
int updateVoxelsTimeSlices(Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, int32_t Iter, uint8_t** Mask);

/*computes the location of (i,j,k) th element in a 1D array*/
int32_t array_loc_1D (int32_t i, int32_t j, int32_t k, int32_t N_j, int32_t N_k)
{
  return (i*N_j*N_k + j*N_k + k);
}

/*computes the value of cost function. 'ErrorSino' is the error sinogram*/
Real_t computeCost(Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr)
{
  Real_t cost=0, temp=0, forward=0, prior=0, detdist_r;
  int32_t i,j,k,p,sino_idx,slice;
 
  AMatrixCol AMatrixPtr_X, AMatrixPtr_Y;
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ObjPtr->delta_x/SinoPtr->delta_r);
  uint8_t AvgNumYElements = (uint8_t)ceil(3*ObjPtr->delta_y/SinoPtr->delta_r);

  Real_arr_t ***ErrSino_Flip_x, ***ErrSino_Flip_y, ***ErrSino_Unflip_x, ***ErrSino_Unflip_y;

  AMatrixPtr_X.values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
  AMatrixPtr_X.index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  AMatrixPtr_Y.values = (Real_t*)get_spc(AvgNumYElements, sizeof(Real_t));
  AMatrixPtr_Y.index = (int32_t*)get_spc(AvgNumYElements, sizeof(int32_t));
  
  ErrSino_Unflip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
  ErrSino_Unflip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
  memset(&(ErrSino_Unflip_x[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));
  memset(&(ErrSino_Unflip_y[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));

#ifdef VFET_ELEC_RECON  
  ErrSino_Flip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
  ErrSino_Flip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
  memset(&(ErrSino_Flip_x[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));
  memset(&(ErrSino_Flip_y[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));
#endif
  
/*  #pragma omp parallel for private(j, k, sino_idx, slice)*/
    for (slice=0; slice<ObjPtr->N_z; slice++){
    for (j=0; j<ObjPtr->N_y; j++)
    {
      for (k=0; k<ObjPtr->N_x; k++){
        for (sino_idx=0; sino_idx < SinoPtr->N_p; sino_idx++){
		detdist_r = (ObjPtr->y0 + ((Real_t)j+0.5)*ObjPtr->delta_y)*SinoPtr->cosine_x[sino_idx];
		detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_x[sino_idx];
		calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse[sino_idx], &(AMatrixPtr_X), detdist_r);
		
		detdist_r = (ObjPtr->x0 + ((Real_t)k+0.5)*ObjPtr->delta_x)*SinoPtr->cosine_y[sino_idx];
		detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_y[sino_idx];
          	calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse[sino_idx], &(AMatrixPtr_Y), detdist_r);
            /*	printf("count = %d, idx = %d, val = %f\n", VoxelLineResponse[slice].count, VoxelLineResponse[slice].index[0], VoxelLineResponse[slice].values[0]);*/

		mag_forward_project_voxel (SinoPtr, InpPtr, ObjPtr->MagPotentials[slice][j][k][0], ObjPtr->MagPotentials[slice][j][k][1], ErrSino_Unflip_x, ErrSino_Flip_x, &(AMatrixPtr_X), &(ObjPtr->VoxelLineResp_X[k]), sino_idx, SinoPtr->cosine_x[sino_idx], SinoPtr->sine_x[sino_idx]);
		mag_forward_project_voxel (SinoPtr, InpPtr, ObjPtr->MagPotentials[slice][j][k][0], ObjPtr->MagPotentials[slice][j][k][2], ErrSino_Unflip_y, ErrSino_Flip_y, &(AMatrixPtr_Y), &(ObjPtr->VoxelLineResp_Y[j]), sino_idx, SinoPtr->cosine_y[sino_idx], SinoPtr->sine_y[sino_idx]);

          }
        }
      }
    }
  
  free(AMatrixPtr_X.values);
  free(AMatrixPtr_X.index);
  free(AMatrixPtr_Y.values);
  free(AMatrixPtr_Y.index);
  
  #pragma omp parallel for private(j, k, temp) reduction(+:forward)
  for (i = 0; i < SinoPtr->N_p; i++)
  for (j = 0; j < SinoPtr->N_r; j++)
  for (k = 0; k < SinoPtr->N_t; k++)
  {
    temp = (SinoPtr->Data_Unflip_x[i][j][k] - ErrSino_Unflip_x[i][j][k]);
    forward += temp*temp*InpPtr->Weight;
    temp = (SinoPtr->Data_Unflip_y[i][j][k] - ErrSino_Unflip_y[i][j][k]);
    forward += temp*temp*InpPtr->Weight;
  }
  
  forward /= 2.0; 
 
  multifree(ErrSino_Unflip_x, 3);
  multifree(ErrSino_Unflip_y, 3);
  /*When computing the cost of the prior term it is important to make sure that you don't include the cost of any pair of neighbors more than once. In this code, a certain sense of causality is used to compute the cost. We also assume that the weghting kernel given by 'Filter' is symmetric. Let i, j and k correspond to the three dimensions. If we go forward to i+1, then all neighbors at j-1, j, j+1, k+1, k, k-1 are to be considered. However, if for the same i, if we go forward to j+1, then all k-1, k, and k+1 should be considered. For same i and j, only the neighbor at k+1 is considred.*/
  prior = 0;
  
  for (p = 0; p < ObjPtr->N_z; p++)
  for (j = 0; j < ObjPtr->N_y; j++)
  for (k = 0; k < ObjPtr->N_x; k++)
  {
	temp = ObjPtr->ErrorPotMag[p][j][k][0]*ObjPtr->ErrorPotMag[p][j][k][0];
	temp += ObjPtr->ErrorPotMag[p][j][k][1]*ObjPtr->ErrorPotMag[p][j][k][1];
	temp += ObjPtr->ErrorPotMag[p][j][k][2]*ObjPtr->ErrorPotMag[p][j][k][2];
	prior += InpPtr->ADMM_mu*temp/2;
  }

    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Forward cost = %f\n",forward);
    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Prior cost = %f\n",prior);
    InpPtr->Forward_Cost = forward;
    InpPtr->Prior_Cost = prior;
    cost = forward + prior;
  
  return cost;
}

/*computes the value of cost function. 'ErrorSino' is the error sinogram*/
Real_t compute_orig_cost(Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr)
{
  Real_t cost=0, temp=0, forward=0, prior=0, detdist_r;
  Real_t Diff;
  Real_arr_t ***ErrSino_Flip_x, ***ErrSino_Flip_y, ***ElecPotentials;
  int32_t p,i,j,k,cidx,sino_idx,slice;
  bool j_minus, k_minus, j_plus, k_plus, p_plus;
 
  AMatrixCol AMatrixPtr_X, AMatrixPtr_Y;
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ObjPtr->delta_x/SinoPtr->delta_r);
  uint8_t AvgNumYElements = (uint8_t)ceil(3*ObjPtr->delta_y/SinoPtr->delta_r);

  AMatrixPtr_X.values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
  AMatrixPtr_X.index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  AMatrixPtr_Y.values = (Real_t*)get_spc(AvgNumYElements, sizeof(Real_t));
  AMatrixPtr_Y.index = (int32_t*)get_spc(AvgNumYElements, sizeof(int32_t));
  
  Real_arr_t*** ErrSino_Unflip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
  Real_arr_t*** ErrSino_Unflip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
  Real_arr_t**** MagPotentials = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);

  memset(&(ErrSino_Unflip_x[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));
  memset(&(ErrSino_Unflip_y[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));
  memset(&(MagPotentials[0][0][0][0]), 0, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*3*sizeof(Real_arr_t));

  compute_magcrossprodtran (ObjPtr->Magnetization, MagPotentials, ObjPtr->MagFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 1);

/*  #pragma omp parallel for private(j, k, sino_idx, slice)*/
    for (slice=0; slice<ObjPtr->N_z; slice++){
    for (j=0; j<ObjPtr->N_y; j++)
    {
      for (k=0; k<ObjPtr->N_x; k++){
        for (sino_idx=0; sino_idx < SinoPtr->N_p; sino_idx++){
		detdist_r = (ObjPtr->y0 + ((Real_t)j+0.5)*ObjPtr->delta_y)*SinoPtr->cosine_x[sino_idx];
		detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_x[sino_idx];
		calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse[sino_idx], &(AMatrixPtr_X), detdist_r);
		
		detdist_r = (ObjPtr->x0 + ((Real_t)k+0.5)*ObjPtr->delta_x)*SinoPtr->cosine_y[sino_idx];
		detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_y[sino_idx];
          	calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse[sino_idx], &(AMatrixPtr_Y), detdist_r);
          	
            /*	printf("count = %d, idx = %d, val = %f\n", VoxelLineResponse[slice].count, VoxelLineResponse[slice].index[0], VoxelLineResponse[slice].values[0]);*/

		mag_forward_project_voxel (SinoPtr, InpPtr, MagPotentials[slice][j][k][0], MagPotentials[slice][j][k][1], ErrSino_Unflip_x, ErrSino_Flip_x, &(AMatrixPtr_X), &(ObjPtr->VoxelLineResp_X[k]), sino_idx, SinoPtr->cosine_x[sino_idx], SinoPtr->sine_x[sino_idx]);
		mag_forward_project_voxel (SinoPtr, InpPtr, MagPotentials[slice][j][k][0], MagPotentials[slice][j][k][2], ErrSino_Unflip_y, ErrSino_Flip_y, &(AMatrixPtr_Y), &(ObjPtr->VoxelLineResp_Y[j]), sino_idx, SinoPtr->cosine_y[sino_idx], SinoPtr->sine_y[sino_idx]);
          }
        }
      }
    }
  
  free(AMatrixPtr_X.values);
  free(AMatrixPtr_X.index);
  free(AMatrixPtr_Y.values);
  free(AMatrixPtr_Y.index);
  
  #pragma omp parallel for private(j, k, temp) reduction(+:forward)
  for (i = 0; i < SinoPtr->N_p; i++)
  for (j = 0; j < SinoPtr->N_r; j++)
  for (k = 0; k < SinoPtr->N_t; k++)
  {
    temp = (SinoPtr->Data_Unflip_x[i][j][k] - ErrSino_Unflip_x[i][j][k]);
    forward += temp*temp*InpPtr->Weight;
    temp = (SinoPtr->Data_Unflip_y[i][j][k] - ErrSino_Unflip_y[i][j][k]);
    forward += temp*temp*InpPtr->Weight;
  }
  forward /= 2.0;
  
  multifree(ErrSino_Unflip_x, 3);
  multifree(ErrSino_Unflip_y, 3);
  multifree(MagPotentials, 4);

  /*When computing the cost of the prior term it is important to make sure that you don't include the cost of any pair of neighbors more than once. In this code, a certain sense of causality is used to compute the cost. We also assume that the weghting kernel given by 'Filter' is symmetric. Let i, j and k correspond to the three dimensions. If we go forward to i+1, then all neighbors at j-1, j, j+1, k+1, k, k-1 are to be considered. However, if for the same i, if we go forward to j+1, then all k-1, k, and k+1 should be considered. For same i and j, only the neighbor at k+1 is considred.*/
  prior = 0;
  #pragma omp parallel for private(Diff, p, j, k, j_minus, k_minus, p_plus, j_plus, k_plus, cidx) reduction(+:prior)
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
      }
      if(j_plus == true) {
        if(k_minus == true) {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = (ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p][j + 1][k - 1][cidx]);
          	prior += InpPtr->Spatial_Filter[1][2][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
          }
        }
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = (ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p][j + 1][k][cidx]);
        	prior += InpPtr->Spatial_Filter[1][2][1] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	}
        if(k_plus == true) {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = (ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p][j + 1][k + 1][cidx]);
          	prior += InpPtr->Spatial_Filter[1][2][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
          }
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
        }
        
	for (cidx = 0; cidx < 3; cidx++){
        	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p+1][j][k][cidx];
        	prior += InpPtr->Spatial_Filter[2][1][1] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	}
        if(j_plus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p+1][j + 1][k][cidx];
          	prior += InpPtr->Spatial_Filter[2][2][1] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	  }
        }
        if(j_minus == true)
        {
          if(k_minus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j - 1][k - 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][0][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
          }
          if(k_plus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j - 1][k + 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][0][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
          }
        }
        if(k_minus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j][k - 1][cidx];
          	prior += InpPtr->Spatial_Filter[2][1][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	  }
        }
        if(j_plus == true)
        {
          if(k_minus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j + 1][k - 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][2][0] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
          }
          if(k_plus == true)
          {
	    for (cidx = 0; cidx < 3; cidx++){
            	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j + 1][k + 1][cidx];
            	prior += InpPtr->Spatial_Filter[2][2][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	    }
          }
        }
        if(k_plus == true)
        {
	  for (cidx = 0; cidx < 3; cidx++){
          	Diff = ObjPtr->Magnetization[p][j][k][cidx] - ObjPtr->Magnetization[p + 1][j][k + 1][cidx];
          	prior += InpPtr->Spatial_Filter[2][1][2] * QGGMRF_Value(Diff,InpPtr->Mag_Sigma_Q[cidx], InpPtr->Mag_Sigma_Q_P[cidx], ObjPtr->Mag_C[cidx]);
	  }
        }
      }
    }
  }
    
   check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Original Forward cost = %f\n",forward);
    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Original Prior cost = %f\n",prior);
    cost = forward + prior;
  
  return cost;
}


/*randomly select the voxels lines which need to be updated along the x-y plane for each z-block and time slice*/
void randomly_select_x_y (ScannedObject* ObjPtr, TomoInputs* InpPtr)
{
  int64_t j, num, n, Index, col, row, *Counter, ArraySize, block, xidx, yidx, zidx;
  ArraySize = ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x;
  Counter = (int64_t*)get_spc(ArraySize, sizeof(int64_t));
    
  for (Index = 0; Index < ArraySize; Index++)
  	Counter[Index] = Index;
    
    InpPtr->UpdateSelectNum = 0;
    for (j=0; j<ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x; j++){
      Index = floor(random2() * ArraySize);
      Index = (Index == ArraySize)? ArraySize-1: Index;
      xidx = Counter[Index] % ObjPtr->N_x;
      yidx = (Counter[Index] / ObjPtr->N_x) % ObjPtr->N_y;
      zidx = (Counter[Index] / (ObjPtr->N_x*ObjPtr->N_y));
        
      num = InpPtr->UpdateSelectNum;
      InpPtr->x_rand_select[num] = xidx;
      InpPtr->y_rand_select[num] = yidx;
      InpPtr->z_rand_select[num] = zidx;

      (InpPtr->UpdateSelectNum)++;
      
      Counter[Index] = Counter[ArraySize - 1];
      ArraySize--;
    }
   
  if (InpPtr->UpdateSelectNum != ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x)
      	check_warn(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Number of voxels to update does not equal the total number of voxels.\n");
	 
 
  free(Counter);
}



/*'initErrorSinogram' is used to initialize the error sinogram before start of ICD. It computes e = y - Ax - d. Ax is computed by forward projecting the object x.*/
int32_t initErrorSinogam (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr)
{
 
  Real_arr_t*** ErrorSino_Unflip_x = SinoPtr->ErrorSino_Unflip_x;
  Real_arr_t*** ErrorSino_Unflip_y = SinoPtr->ErrorSino_Unflip_y;
  Real_arr_t*** ErrorSino_Flip_x = SinoPtr->ErrorSino_Flip_x;
  Real_arr_t*** ErrorSino_Flip_y = SinoPtr->ErrorSino_Flip_y;
  
  Real_t unflipavg = 0, flipavg = 0, potxavg = 0, potyavg = 0, potzavg = 0, potrhoavg = 0, detdist_r;
  int32_t i, j, k, sino_idx, slice, flag = 0;
  AMatrixCol AMatrixPtr_X, AMatrixPtr_Y;
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ObjPtr->delta_x/SinoPtr->delta_r);
  uint8_t AvgNumYElements = (uint8_t)ceil(3*ObjPtr->delta_y/SinoPtr->delta_r);
 /* char error_file[100];*/

  AMatrixPtr_X.values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
  AMatrixPtr_X.index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
  AMatrixPtr_Y.values = (Real_t*)get_spc(AvgNumYElements, sizeof(Real_t));
  AMatrixPtr_Y.index = (int32_t*)get_spc(AvgNumYElements, sizeof(int32_t));
  
  memset(&(ErrorSino_Unflip_x[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));
  memset(&(ErrorSino_Unflip_y[0][0][0]), 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(Real_arr_t));
  memset(&(ObjPtr->MagPotentials[0][0][0][0]), 0, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*3*sizeof(Real_arr_t));
  memset(&(ObjPtr->ErrorPotMag[0][0][0][0]), 0, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*3*sizeof(Real_arr_t));
  memset(&(ObjPtr->MagPotDual[0][0][0][0]), 0, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*3*sizeof(Real_arr_t));

  compute_magcrossprodtran (ObjPtr->Magnetization, ObjPtr->MagPotentials, ObjPtr->MagFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 1);

  for (i = 0; i < ObjPtr->N_z; i++)
  for (j = 0; j < ObjPtr->N_y; j++)
  for (k = 0; k < ObjPtr->N_x; k++)
  {
	potzavg += ObjPtr->MagPotentials[i][j][k][0];
	potyavg += ObjPtr->MagPotentials[i][j][k][1];
	potxavg += ObjPtr->MagPotentials[i][j][k][2];
  }	

  potzavg /= (ObjPtr->N_x*ObjPtr->N_y*ObjPtr->N_z);
  potyavg /= (ObjPtr->N_x*ObjPtr->N_y*ObjPtr->N_z);
  potxavg /= (ObjPtr->N_x*ObjPtr->N_y*ObjPtr->N_z);
  check_debug(InpPtr->node_rank == 0, InpPtr->debug_file_ptr, "Average of potentials after forward projection are (x, y, z) = (%f, %f, %f)\n", potxavg, potyavg, potzavg);

/*  #pragma omp parallel for private(j, k, sino_idx, slice)*/
    for (slice=0; slice<ObjPtr->N_z; slice++){
    for (j=0; j<ObjPtr->N_y; j++)
    {
      for (k=0; k<ObjPtr->N_x; k++){
        for (sino_idx=0; sino_idx < SinoPtr->N_p; sino_idx++){
		detdist_r = (ObjPtr->y0 + ((Real_t)j+0.5)*ObjPtr->delta_y)*SinoPtr->cosine_x[sino_idx];
		detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_x[sino_idx];
		calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse[sino_idx], &(AMatrixPtr_X), detdist_r);
		
		detdist_r = (ObjPtr->x0 + ((Real_t)k+0.5)*ObjPtr->delta_x)*SinoPtr->cosine_y[sino_idx];
		detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_y[sino_idx];
          	calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse[sino_idx], &(AMatrixPtr_Y), detdist_r);
    	      	
            /*	printf("count = %d, idx = %d, val = %f\n", VoxelLineResponse[slice].count, VoxelLineResponse[slice].index[0], VoxelLineResponse[slice].values[0]);*/
		mag_forward_project_voxel (SinoPtr, InpPtr, ObjPtr->MagPotentials[slice][j][k][0], ObjPtr->MagPotentials[slice][j][k][1], ErrorSino_Unflip_x, ErrorSino_Flip_x, &(AMatrixPtr_X), &(ObjPtr->VoxelLineResp_X[k]), sino_idx, SinoPtr->cosine_x[sino_idx], SinoPtr->sine_x[sino_idx]);
		mag_forward_project_voxel (SinoPtr, InpPtr, ObjPtr->MagPotentials[slice][j][k][0], ObjPtr->MagPotentials[slice][j][k][2], ErrorSino_Unflip_y, ErrorSino_Flip_y, &(AMatrixPtr_Y), &(ObjPtr->VoxelLineResp_Y[j]), sino_idx, SinoPtr->cosine_y[sino_idx], SinoPtr->sine_y[sino_idx]);
          }
        }
      }
    }
  

  #pragma omp parallel for private(j, k) reduction(+:unflipavg,flipavg)
  for(i = 0; i < SinoPtr->N_p; i++)
  for(j = 0; j < SinoPtr->N_r; j++)
  for(k = 0; k < SinoPtr->N_t; k++)
  {
    	unflipavg += ErrorSino_Unflip_x[i][j][k];
    	unflipavg += ErrorSino_Unflip_y[i][j][k];
	ErrorSino_Unflip_x[i][j][k] = SinoPtr->Data_Unflip_x[i][j][k] - ErrorSino_Unflip_x[i][j][k];
	ErrorSino_Unflip_y[i][j][k] = SinoPtr->Data_Unflip_y[i][j][k] - ErrorSino_Unflip_y[i][j][k];
  }
  unflipavg = unflipavg/(SinoPtr->N_r*SinoPtr->N_t*SinoPtr->N_p);
  check_debug(InpPtr->node_rank == 0, InpPtr->debug_file_ptr, "Average of unflipped component of forward projection in node %d is %f\n", InpPtr->node_rank, unflipavg);

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
  int updateVoxelsTimeSlices(Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, int32_t Iter, uint8_t** Mask)
  {
    Real_t /*AverageUpdate = 0, tempUpdate, avg_update_percentage, total_vox_mag = 0.0, vox_mag = 0.0, */magpot_update = 0, elecpot_update = 0, magpot_sum = 0, elecpot_sum = 0, magpot_update_tot = 0, elecpot_update_tot = 0, magpot_sum_tot = 0, elecpot_sum_tot = 0;
    int32_t xyz_start, xyz_end, j, K, block, idx;
/*    Real_t tempTotPix = 0, total_pix = 0;*/
    /*MPI_Request mag_send_reqs, mag_recv_reqs, elec_send_reqs, elec_recv_reqs;*/
    
    randomly_select_x_y (ObjPtr, InpPtr);
    
    /*	K = ObjPtr->N_time*ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x;
    K = (K - total_zero_count)/(ObjPtr->gamma*K);*/
    K = ObjPtr->NHICD_Iterations;
    check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Number of NHICD iterations is %d.\n", K);
    for (j = 0; j < K; j++)
    {
/*      total_vox_mag = 0.0;*/
      /*#pragma omp parallel for private(block, idx, xy_start, xy_end) reduction(+:total_vox_mag)*/
        magpot_update = 0; magpot_sum = 0;
        xyz_start = j*floor(InpPtr->UpdateSelectNum/K);
        xyz_end = (j + 1)*floor(InpPtr->UpdateSelectNum/K) - 1;
        xyz_end = (j == K - 1) ? InpPtr->UpdateSelectNum - 1: xyz_end;
        /*	printf ("Loop 1 Start - j = %d, i = %d, idx = %d, z_start = %d, z_stop = %d, xy_start = %d, xy_end = %d\n", j, i, idx, z_start[i][idx], z_stop[i][idx], xy_start, xy_end);*/
        updateVoxels (xyz_start, xyz_end, InpPtr->x_rand_select, InpPtr->y_rand_select, InpPtr->z_rand_select, SinoPtr, ObjPtr, InpPtr, SinoPtr->ErrorSino_Unflip_x, SinoPtr->ErrorSino_Flip_x, SinoPtr->ErrorSino_Unflip_y, SinoPtr->ErrorSino_Flip_y, SinoPtr->DetectorResponse, /*VoxelLineResponse,*/ Iter, ObjPtr->MagPotUpdateMap, &magpot_update, &magpot_sum, Mask);

	magpot_update_tot += magpot_update;
	magpot_sum_tot += magpot_sum;
      
      /*MPI_Send_Recv_Z_Slices (ObjPtr, InpPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 1);*/
/*      MPI_Wait_Z_Slices (ObjPtr, InpPtr, &mag_send_reqs, &elec_send_reqs, &mag_recv_reqs, &elec_recv_reqs, 1);*/
      
      VSC_based_Voxel_Line_Select(ObjPtr, InpPtr, ObjPtr->MagPotUpdateMap);
      if (Iter > 1 && InpPtr->no_NHICD == 0)
      {
        /*#pragma omp parallel for private(block, idx)*/
          updateVoxels (0, InpPtr->NHICDSelectNum-1, InpPtr->x_NHICD_select, InpPtr->y_NHICD_select, InpPtr->z_NHICD_select, SinoPtr, ObjPtr, InpPtr, SinoPtr->ErrorSino_Unflip_x, SinoPtr->ErrorSino_Flip_x, SinoPtr->ErrorSino_Unflip_y, SinoPtr->ErrorSino_Flip_y, SinoPtr->DetectorResponse, Iter, ObjPtr->MagPotUpdateMap, &magpot_update, &magpot_sum, Mask);  
      }
    }
    
    /*MPI_Allreduce(&AverageUpdate, &tempUpdate, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_pix, &tempTotPix, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_vox_mag, &vox_mag, 1, MPI_REAL_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    AverageUpdate = tempUpdate/(tempTotPix);
    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Average voxel update over all voxels is %e, total voxels is %e.\n", AverageUpdate, tempTotPix);
    */
    /*	multifree(offset_numerator,2);
    multifree(offset_denominator,2);*/
/*    avg_update_percentage = 100*tempUpdate/vox_mag;*/

    magpot_update = 100*magpot_update_tot/magpot_sum_tot;
    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Percentage average magnitude of voxel updates of (Magnetic) potential is (%e).\n", magpot_update);
   
    if (magpot_update < InpPtr->StopThreshold)
    {
      check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Percentage average magnitude of voxel updates is less than convergence threshold.\n");
      return (1);
    }

    return(0);
  }

  /*ICD_BackProject calls the ICD optimization function repeatedly till the stopping criteria is met.*/
  int ICD_BackProject(Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr)
  {
    #ifndef NO_COST_CALCULATE
    Real_t cost, cost_0_iter, cost_last_iter, percentage_change_in_cost = 0, orig_cost_last = 0, orig_cost = 0;
    char costfile[100] = COST_FILENAME, origcostfile[100] = ORIG_COST_FILENAME;
    #endif
    Real_t x, y, DualMag[3], DualElec, mag_primal_res = 0, elec_primal_res = 0;
    int32_t t, i, j, flag = 0, Iter, k, HeadIter;
    int dimTiff[4];
    time_t start;
    char detect_file[100] = DETECTOR_RESPONSE_FILENAME;
    /*char MagPotUpdateMapFile[100] = MAGPOT_UPDATE_MAP_FILENAME;
    char ElecPotUpdateMapFile[100] = ELECPOT_UPDATE_MAP_FILENAME;*/
    uint8_t **Mask;

    SinoPtr->ZLineResponse = (Real_arr_t *)get_spc(DETECTOR_RESPONSE_BINS + 1, sizeof(Real_arr_t));
    ZLineResponseProfile (SinoPtr, ObjPtr, InpPtr);
    
    ObjPtr->VoxelLineResp_X = (AMatrixCol*)get_spc(ObjPtr->N_x, sizeof(AMatrixCol));
    uint8_t AvgNumXElements = (uint8_t)((ObjPtr->delta_x/SinoPtr->delta_t) + 2);
    for (t = 0; t < ObjPtr->N_x; t++){
    	ObjPtr->VoxelLineResp_X[t].values = (Real_t*)get_spc(AvgNumXElements, sizeof(Real_t));
    	ObjPtr->VoxelLineResp_X[t].index = (int32_t*)get_spc(AvgNumXElements, sizeof(int32_t));
    }
    storeVoxelLineResponse(ObjPtr->VoxelLineResp_X, SinoPtr, ObjPtr->x0, ObjPtr->delta_x, ObjPtr->N_x);
    
    ObjPtr->VoxelLineResp_Y = (AMatrixCol*)get_spc(ObjPtr->N_y, sizeof(AMatrixCol));
    uint8_t AvgNumYElements = (uint8_t)((ObjPtr->delta_y/SinoPtr->delta_t) + 2);
    for (t = 0; t < ObjPtr->N_y; t++){
    	ObjPtr->VoxelLineResp_Y[t].values = (Real_t*)get_spc(AvgNumYElements, sizeof(Real_t));
    	ObjPtr->VoxelLineResp_Y[t].index = (int32_t*)get_spc(AvgNumYElements, sizeof(int32_t));
    }
    storeVoxelLineResponse(ObjPtr->VoxelLineResp_Y, SinoPtr, ObjPtr->y0, ObjPtr->delta_y, ObjPtr->N_y);
    

    #ifdef POSITIVITY_CONSTRAINT
    	check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Enforcing positivity constraint\n");
    #endif
    
/*    ObjPtr->MagPotUpdateMap = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, InpPtr->num_z_blocks, ObjPtr->N_y, ObjPtr->N_x);
    ObjPtr->ElecPotUpdateMap = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, InpPtr->num_z_blocks, ObjPtr->N_y, ObjPtr->N_x);*/
    SinoPtr->DetectorResponse = (Real_arr_t **)multialloc(sizeof(Real_arr_t), 2, SinoPtr->N_p, DETECTOR_RESPONSE_BINS + 1);

    SinoPtr->ErrorSino_Unflip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
    SinoPtr->ErrorSino_Unflip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
    ObjPtr->ErrorPotMag = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
   
    Mask = (uint8_t**)multialloc(sizeof(uint8_t), 2, ObjPtr->N_y, ObjPtr->N_x);
    
/*    memset(&(ObjPtr->MagPotUpdateMap[0][0][0]), 0, InpPtr->num_z_blocks*ObjPtr->N_y*ObjPtr->N_x*sizeof(Real_arr_t));
    memset(&(ObjPtr->ElecPotUpdateMap[0][0][0]), 0, InpPtr->num_z_blocks*ObjPtr->N_y*ObjPtr->N_x*sizeof(Real_arr_t));*/
/*    omp_set_num_threads(InpPtr->num_threads);*/
    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Number of CPU cores is %d\n", (int)omp_get_num_procs());
    /*	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ICD_BackProject: Number of threads is %d\n", InpPtr->num_threads) ;*/
    for (j = 0; j < ObjPtr->N_y; j++)
    for (k = 0; k < ObjPtr->N_x; k++){
      x = ObjPtr->x0 + ((Real_t)k + 0.5)*ObjPtr->delta_x;
      y = ObjPtr->y0 + ((Real_t)j + 0.5)*ObjPtr->delta_y;
      if (x*x + y*y < InpPtr->radius_obj*InpPtr->radius_obj)
        Mask[j][k] = 1;
      else
        Mask[j][k] = 0;
    }
    
    DetectorResponseProfile (SinoPtr, ObjPtr, InpPtr);
    dimTiff[0] = 1; dimTiff[1] = 1; dimTiff[2] = SinoPtr->N_p; dimTiff[3] = DETECTOR_RESPONSE_BINS+1;
    sprintf(detect_file, "%s_n%d", detect_file, InpPtr->node_rank);
    if (InpPtr->Write2Tiff == 1)
    	if (WriteMultiDimArray2Tiff (detect_file, dimTiff, 0, 1, 2, 3, &(SinoPtr->DetectorResponse[0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
    start = time(NULL);
    
    if (initObject(SinoPtr, ObjPtr, InpPtr)) goto error;
    if (initErrorSinogam(SinoPtr, ObjPtr, InpPtr, fftptr)) goto error;
/*    if (init_minmax_object (ObjPtr, InpPtr)) goto error;*/

    check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Time taken to initialize object and compute error sinogram = %fmins\n", difftime(time(NULL),start)/60.0);
  
    start=time(NULL);
    
    orig_cost_last = compute_orig_cost(SinoPtr, ObjPtr, InpPtr, fftptr);
    check_info(InpPtr->node_rank == 0, InpPtr->debug_file_ptr, "HeadIter = 0: The original cost value is %f.\n", orig_cost_last);
    if (InpPtr->node_rank == 0)
	   Write2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost_last, InpPtr->debug_file_ptr);

    for (HeadIter = 1; HeadIter <= InpPtr->Head_MaxIter; HeadIter++)
    {
	    reconstruct_magnetization(ObjPtr, InpPtr, fftptr);

#ifndef NO_COST_CALCULATE
	    cost = computeCost(SinoPtr,ObjPtr,InpPtr);
	    cost_0_iter = cost;
	    cost_last_iter = cost;
	    check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "------------- Iteration 0, Cost = %f------------\n",cost);
	    if (InpPtr->node_rank == 0)
	    	Write2Bin (costfile, 1, 1, 1, 1, sizeof(Real_t), &cost, InpPtr->debug_file_ptr);
#endif /*Cost calculation endif*/
		
    	for (Iter = 1; Iter <= InpPtr->NumIter; Iter++)
    	{
      		flag = updateVoxelsTimeSlices (SinoPtr, ObjPtr, InpPtr, Iter, Mask);
      		if (InpPtr->WritePerIter == 1)
      			if (write_ObjectProjOff2TiffBinPerIter (SinoPtr, ObjPtr, InpPtr)) goto error;
#ifndef NO_COST_CALCULATE
	      cost = computeCost(SinoPtr,ObjPtr,InpPtr);
	      percentage_change_in_cost = ((cost - cost_last_iter)/(cost - cost_0_iter))*100.0;
	      check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Percentage change in cost is %f.\n", percentage_change_in_cost);
	      check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "------------- Iteration = %d, Cost = %f, Time since start of ICD = %fmins ------------\n",Iter,cost,difftime(time(NULL),start)/60.0);
	      if (InpPtr->node_rank == 0)
			Append2Bin (costfile, 1, 1, 1, 1, sizeof(Real_t), &cost, InpPtr->debug_file_ptr);
	      
	      if (cost > cost_last_iter)
		      check_info(InpPtr->node_rank == 0, InpPtr->debug_file_ptr, "ERROR: Cost value increased.\n");
	      cost_last_iter = cost;
	      /*if (percentage_change_in_cost < InpPtr->cost_thresh && flag != 0 && Iter > 1){*/
	      if (flag != 0 && Iter > 1){
		        check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Convergence criteria is met.\n");
        		break;
      		}
#else
	      check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "-------------ICD_BackProject: ICD Iter = %d, time since start of ICD = %fmins------------.\n",Iter,difftime(time(NULL),start)/60.0);
		if (flag != 0 && Iter > 1){
        		check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Convergence criteria is met.\n");
        		break;
	      }
#endif
	      flag = fflush(InpPtr->debug_file_ptr);
     		 if (flag != 0)
      			check_warn(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Cannot flush buffer.\n");
    	}

	mag_primal_res = 0; elec_primal_res = 0;
	for (i = 0; i < ObjPtr->N_z; i++)
	for (j = 0; j < ObjPtr->N_y; j++)
	for (k = 0; k < ObjPtr->N_x; k++)
	{
		DualMag[0] = ObjPtr->MagPotDual[i][j][k][0];
		DualMag[1] = ObjPtr->MagPotDual[i][j][k][1];
		DualMag[2] = ObjPtr->MagPotDual[i][j][k][2];
		
		ObjPtr->MagPotDual[i][j][k][0] = -ObjPtr->ErrorPotMag[i][j][k][0];
		ObjPtr->MagPotDual[i][j][k][1] = -ObjPtr->ErrorPotMag[i][j][k][1];
		ObjPtr->MagPotDual[i][j][k][2] = -ObjPtr->ErrorPotMag[i][j][k][2];
	
		ObjPtr->ErrorPotMag[i][j][k][0] -= (ObjPtr->MagPotDual[i][j][k][0] - DualMag[0]);	
		ObjPtr->ErrorPotMag[i][j][k][1] -= (ObjPtr->MagPotDual[i][j][k][1] - DualMag[1]);	
		ObjPtr->ErrorPotMag[i][j][k][2] -= (ObjPtr->MagPotDual[i][j][k][2] - DualMag[2]);

		mag_primal_res += fabs(ObjPtr->ErrorPotMag[i][j][k][0] + ObjPtr->MagPotDual[i][j][k][0]);  
		mag_primal_res += fabs(ObjPtr->ErrorPotMag[i][j][k][1] + ObjPtr->MagPotDual[i][j][k][1]);  
		mag_primal_res += fabs(ObjPtr->ErrorPotMag[i][j][k][2] + ObjPtr->MagPotDual[i][j][k][2]);  
	
	}

        orig_cost = compute_orig_cost(SinoPtr, ObjPtr, InpPtr, fftptr);
        check_info(InpPtr->node_rank == 0, InpPtr->debug_file_ptr, "HeadIter = %d: The original cost value is %f. The decrease in original cost is %f.\n", HeadIter, orig_cost, orig_cost_last - orig_cost);
	
	mag_primal_res /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
        check_info(InpPtr->node_rank == 0, InpPtr->debug_file_ptr, "Mag average primal residual is %e.\n", mag_primal_res);
	
    	if (InpPtr->node_rank == 0)
	   Append2Bin (origcostfile, 1, 1, 1, 1, sizeof(Real_t), &orig_cost, InpPtr->debug_file_ptr);
	
	if (orig_cost > orig_cost_last)
      		check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "WARNING: Cost of original cost function increased!\n");
	orig_cost_last = orig_cost;
	
    	if (write_ObjectProjOff2TiffBinPerIter (SinoPtr, ObjPtr, InpPtr)) goto error;

	/*if (avg_head_update < InpPtr->Head_threshold && HeadIter > 1)
		break;*/
    }

 /*   int32_t size = InpPtr->num_z_blocks*ObjPtr->N_y*ObjPtr->N_x;
    if (write_SharedBinFile_At (MagPotUpdateMapFile, &(ObjPtr->MagPotUpdateMap[0][0][0]), InpPtr->node_rank*size, size, InpPtr->debug_file_ptr)) goto error;
    if (write_SharedBinFile_At (ElecPotUpdateMapFile, &(ObjPtr->ElecPotUpdateMap[0][0][0]), InpPtr->node_rank*size, size, InpPtr->debug_file_ptr)) goto error;*/
    
    dimTiff[0] = 1; dimTiff[1] = SinoPtr->N_p; dimTiff[2] = SinoPtr->N_r; dimTiff[3] = SinoPtr->N_t;
    if (InpPtr->Write2Tiff == 1)
    {
    	if (WriteMultiDimArray2Tiff (ERRORSINO_UNFLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->ErrorSino_Unflip_x[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
    	if (WriteMultiDimArray2Tiff (ERRORSINO_UNFLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->ErrorSino_Unflip_y[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
    }

    multifree(SinoPtr->ErrorSino_Unflip_x, 3);
    multifree(SinoPtr->ErrorSino_Unflip_y, 3);
    multifree(ObjPtr->ErrorPotMag, 4);

    multifree(SinoPtr->DetectorResponse, 2);
    multifree(Mask, 2);
	
    for (t = 0; t < ObjPtr->N_x; t++){
      	free(ObjPtr->VoxelLineResp_X[t].values);
        free(ObjPtr->VoxelLineResp_X[t].index);
    }
    free(ObjPtr->VoxelLineResp_X);
    
    for (t = 0; t < ObjPtr->N_y; t++){
      	free(ObjPtr->VoxelLineResp_Y[t].values);
        free(ObjPtr->VoxelLineResp_Y[t].index);
    }
    free(ObjPtr->VoxelLineResp_Y);
    free(SinoPtr->ZLineResponse);

    /*multifree(ObjPtr->MagPotUpdateMap, 3);
    multifree(ObjPtr->ElecPotUpdateMap, 3);*/
 
    check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Finished running ICD_BackProject.\n");
    flag = fflush(InpPtr->debug_file_ptr);
    if (flag != 0)
       check_warn(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Cannot flush buffer.\n");
    
    return(0);

error:
    multifree(SinoPtr->ErrorSino_Unflip_x, 3);
    multifree(SinoPtr->ErrorSino_Flip_x, 3);
    multifree(SinoPtr->DetectorResponse,2);
    multifree(Mask,2);
   /* multifree(ObjPtr->MagPotUpdateMap, 3);
    multifree(ObjPtr->ElecPotUpdateMap, 3);*/
    return(-1);
  }

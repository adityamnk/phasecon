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
#include <math.h>
#include <stdlib.h>
#include "allocate.h"
#include "XT_Constants.h"
#include "randlib.h"
#include <getopt.h>
#include "XT_Structures.h"
#include <ctype.h>
#include "XT_IOMisc.h"
#include "XT_MPI.h"
#include "XT_MPIIO.h"
#include "omp.h"
#include "XT_Debug.h"
#include "XT_OffsetError.h"
/*For each time slice in the reconstruction, the function copies the corresponding view indices to a new array (which is then usedafter copying). 
--Inputs--
time - index of time slice
ViewIndex - contains the indices of the views which are assumed to be the forward projections of reconstruction at index 'time'
ViewNum - Number of such views*/
void copyViewIndexMap (ScannedObject* ScannedObjectPtr, int32_t time, int32_t* ViewIndex, int32_t ViewNum)
{
	int32_t i;
        ScannedObjectPtr->ProjIdxPtr[time] = (int32_t*)get_spc(ViewNum, sizeof(int32_t));
	for(i=0; i<ViewNum; i++){
		ScannedObjectPtr->ProjIdxPtr[time][i] = ViewIndex[i];
	}
	ScannedObjectPtr->ProjNum[time] = ViewNum;
}

/*Maps the projections to reconstruction time slices.
A projection at time 'projection_time' is assigned to a certain time slice in the reconstruction
 if the start and end time of the time slice includes the 'projection_time'. */
void initSparseAnglesOfObject(Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	int32_t  i, j, proj_start, *ViewIndex, ViewNum;
	Real_t projection_time; 

	ViewIndex = (int32_t*)get_spc(SinogramPtr->N_p, sizeof(int32_t));
	proj_start = 0;
	for (i=0; i<ScannedObjectPtr->N_time; i++)
	{
		ViewNum = 0;
		for (j=proj_start; j < SinogramPtr->N_p; j++)
		{		
			projection_time = SinogramPtr->TimePtr[j];
			if (projection_time >= ScannedObjectPtr->recon_times[i] && projection_time < ScannedObjectPtr->recon_times[i+1])
			{
				ViewIndex[ViewNum] = j;
				ViewNum++;
			}
			if (projection_time >= ScannedObjectPtr->recon_times[i+1]) 
			{
				proj_start = j;
				break;
			}
		}
		copyViewIndexMap (ScannedObjectPtr, i, ViewIndex, ViewNum);
	}
	free(ViewIndex);
}


/*Populates the Views and times of each projection from the text file view_info.txt into arrays*/
int initRandomAngles (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	int32_t i, sino_view, k;
      	Real_t delta_Rtime = 0;
 
        ScannedObjectPtr->ProjIdxPtr = (int32_t**)get_spc(ScannedObjectPtr->N_time, sizeof(int32_t*));
	ScannedObjectPtr->ProjNum = (int32_t*)get_spc(ScannedObjectPtr->N_time, sizeof(int32_t));
	memset(&(ScannedObjectPtr->ProjNum[0]), 0, ScannedObjectPtr->N_time*sizeof(int32_t));

	for (i=0; i<SinogramPtr->N_p-1; i++)
		check_error(SinogramPtr->TimePtr[i+1] < SinogramPtr->TimePtr[i], TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "Projection acquisition time decreased for increasing index. Check input files.\n");

	for (i=0; i<ScannedObjectPtr->N_time; i++)
		check_error(ScannedObjectPtr->recon_times[i+1] < ScannedObjectPtr->recon_times[i], TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "Reconstruction times decreased for increasing index. Check input files.\n");

	initSparseAnglesOfObject(SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "The initialized angle indices of sinogram as corresponding to the scanned object are ...\n");
	for (i=0; i<ScannedObjectPtr->N_time; i++){
		delta_Rtime += ScannedObjectPtr->ProjNum[i];
		check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Object %d : ", i);
		for (k=0; k<ScannedObjectPtr->ProjNum[i]; k++){
			sino_view = ScannedObjectPtr->ProjIdxPtr[i][k];
			check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "%.1f, ", SinogramPtr->ViewPtr[sino_view]*180/M_PI);
		}
		check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "\n");
	}	
	ScannedObjectPtr->delta_recon = delta_Rtime/ScannedObjectPtr->N_time;

	return(0);
error:
	return(-1);
}

/*Computes the Euclidean distance of a voxel to its neighboring voxels
--Inputs--
i, j, k, l are the indices of neighoring voxels
--Outputs--
Returns the distance */
Real_t distance2node(uint8_t i, uint8_t j, uint8_t k, uint8_t l)
{
	return(sqrt(pow((Real_t)i-1.0, 2.0)+pow((Real_t)j-1.0, 2.0)+pow((Real_t)k-1.0, 2.0)+pow((Real_t)l-1.0, 2.0)));
}

/*Initializes the weights w_{ij} used in the qGGMRF models*/
void initFilter (ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	uint8_t i,j,k;
	Real_t temp1,sum=0,prior_const=0;
/*	prior_const = ScannedObjectPtr->delta_xy*ScannedObjectPtr->delta_xy*ScannedObjectPtr->delta_xy*ScannedObjectPtr->delta_Rtime;*/
	prior_const = ScannedObjectPtr->mult_xy*ScannedObjectPtr->mult_xy*ScannedObjectPtr->mult_xy*ScannedObjectPtr->delta_recon;
/*Filter coefficients of neighboring pixels are inversely proportional to the distance from the center pixel*/
	TomoInputsPtr->Time_Filter[0] = 1.0/distance2node(0,1,1,1);
	sum += 2.0*TomoInputsPtr->Time_Filter[0];
	
	for (i=0; i<NHOOD_Y_MAXDIM; i++)
	for (j=0; j<NHOOD_X_MAXDIM; j++)
	for (k=0; k<NHOOD_Z_MAXDIM; k++){
	if(i!=(NHOOD_Y_MAXDIM)/2 || j!=(NHOOD_X_MAXDIM-1)/2 || k!=(NHOOD_Z_MAXDIM-1)/2)
	{
		temp1 = 1.0/distance2node(1,i,j,k);
		TomoInputsPtr->Spatial_Filter[i][j][k] = temp1;
		sum=sum+temp1;
	}
	else
		TomoInputsPtr->Spatial_Filter[i][j][k]=0;
	}

	for (i=0; i<NHOOD_Y_MAXDIM; i++)
	for (j=0; j<NHOOD_X_MAXDIM; j++)
	for (k=0; k<NHOOD_Z_MAXDIM; k++){
		TomoInputsPtr->Spatial_Filter[i][j][k] = prior_const*TomoInputsPtr->Spatial_Filter[i][j][k]/sum;
	}

	TomoInputsPtr->Time_Filter[0] = prior_const*TomoInputsPtr->Time_Filter[0]/sum;

#ifdef EXTRA_DEBUG_MESSAGES
	sum=0;
	for (i=0; i<NHOOD_Y_MAXDIM; i++)
		for (j=0; j<NHOOD_X_MAXDIM; j++)
			for (k=0; k<NHOOD_Z_MAXDIM; k++)
			{
				sum += TomoInputsPtr->Spatial_Filter[i][j][k];
				check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "initFilter: Filter i=%d, j=%d, k=%d, coeff = %f\n", i,j,k,TomoInputsPtr->Spatial_Filter[i][j][k]/prior_const);
			}
			check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "initFilter: Filter i=0 is %f\n", TomoInputsPtr->Time_Filter[0]/prior_const);
			check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "initFilter: Sum of filter coefficients is %f\n",(sum+2.0*TomoInputsPtr->Time_Filter[0])/prior_const);	
			check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "initFilter: delta_xy*delta_xy*delta_z*delta_tau = %f\n",prior_const);	
#endif /*#ifdef DEBUG_EN*/


}

/*Initializes the sines and cosines of angles at which projections are acquired. It is then used when computing the voxel profile*/
void calculateSinCos(Sinogram* SinogramPtr, TomoInputs* TomoInputsPtr)
{
  int32_t i;

  SinogramPtr->cosine=(Real_t*)get_spc(SinogramPtr->N_p, sizeof(Real_t));
  SinogramPtr->sine=(Real_t*)get_spc(SinogramPtr->N_p, sizeof(Real_t));

  for(i=0;i<SinogramPtr->N_p;i++)
  {
    SinogramPtr->cosine[i]=cos(SinogramPtr->ViewPtr[i]);
    SinogramPtr->sine[i]=sin(SinogramPtr->ViewPtr[i]);
  }
  check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "calculateSinCos: Calculated sines and cosines of angles of rotation\n");
}


/*Initializes the variables in the three major structures used throughout the code -
Sinogram, ScannedObject, TomoInputs. It also allocates memory for several variables.*/
int32_t initStructures (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, int32_t mult_idx, int32_t mult_xy[], int32_t mult_z[], float *measurements, float *weights, float *proj_angles, float *proj_times, float *recon_times, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, int32_t recon_num, Real_t vox_wid, Real_t rot_center, Real_t mag_sig_s, Real_t mag_sig_t, Real_t mag_c_s, Real_t mag_c_t, Real_t phase_sig_s, Real_t phase_sig_t, Real_t phase_c_s, Real_t phase_c_t, Real_t convg_thresh)
{
	int flag = 0, size, i;
	MPI_Comm_size(MPI_COMM_WORLD, &(TomoInputsPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(TomoInputsPtr->node_rank));
	
	ScannedObjectPtr->Mag_Sigma_S = mag_sig_s;
	ScannedObjectPtr->Mag_C_S = mag_c_s;
	ScannedObjectPtr->Mag_Sigma_T = mag_sig_t;
	ScannedObjectPtr->Mag_C_T = mag_c_t;
	
	ScannedObjectPtr->Phase_Sigma_S = phase_sig_s;
	ScannedObjectPtr->Phase_C_S = phase_c_s;
	ScannedObjectPtr->Phase_Sigma_T = phase_sig_t;
	ScannedObjectPtr->Phase_C_T = phase_c_t;
	
	ScannedObjectPtr->mult_xy = mult_xy[mult_idx];
	ScannedObjectPtr->mult_z = mult_z[mult_idx];		
	SinogramPtr->Length_R = vox_wid*proj_cols;
	SinogramPtr->Length_T = vox_wid*proj_rows;
	TomoInputsPtr->StopThreshold = convg_thresh;
	TomoInputsPtr->NumIter = MAX_NUM_ITERATIONS;
	TomoInputsPtr->RotCenter = rot_center;
	TomoInputsPtr->alpha = OVER_RELAXATION_FACTOR;
	if (mult_idx == 0)
		TomoInputsPtr->initICD = 0;
	else if (mult_z[mult_idx] == mult_z[mult_idx-1]) 
		TomoInputsPtr->initICD = 2;
	else if (mult_z[mult_idx-1]/mult_z[mult_idx] == 2)
		TomoInputsPtr->initICD = 3;
	else
		sentinel(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Multi-resolution scaling is not supported");
		
	TomoInputsPtr->Write2Tiff = ENABLE_TIFF_WRITES;
	ScannedObjectPtr->N_time = recon_num;
	SinogramPtr->N_p = proj_num;	
	SinogramPtr->N_r = proj_cols;
	TomoInputsPtr->cost_thresh = COST_CONVG_THRESHOLD;	
	TomoInputsPtr->radius_obj = vox_wid*proj_cols/2;	
	SinogramPtr->total_t_slices = proj_rows;

	TomoInputsPtr->no_NHICD = NO_NHICD;	
	TomoInputsPtr->WritePerIter = WRITE_EVERY_ITER;
		
	TomoInputsPtr->initMagUpMap = 0;
	if (mult_idx > 0)
	{
		TomoInputsPtr->initMagUpMap = 1;
	}

	/*Initializing Sinogram parameters*/
	int32_t j, k, idx;

	SinogramPtr->Length_T = SinogramPtr->Length_T/TomoInputsPtr->node_num;
	SinogramPtr->N_t = SinogramPtr->total_t_slices/TomoInputsPtr->node_num;
	
	SinogramPtr->Measurements = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->MagTomoAux = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t, 4);
	SinogramPtr->MagTomoDual = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->PhaseTomoAux = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t, 4);
	SinogramPtr->PhaseTomoDual = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

	SinogramPtr->MagPRetAux = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->MagPRetDual = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->PhasePRetAux = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->PhasePRetDual = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	TomoInputsPtr->Weight = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

	SinogramPtr->Omega_real = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->Omega_imag = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->D_real = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->D_imag = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

	for (i = 0; i < SinogramPtr->N_p; i++)
	for (j = 0; j < SinogramPtr->N_r; j++)
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		idx = i*SinogramPtr->N_t*SinogramPtr->N_r + k*SinogramPtr->N_r + j;
		SinogramPtr->Measurements[i][j][k] = measurements[idx];		
		TomoInputsPtr->Weight[i][j][k] = weights[idx];

		SinogramPtr->MagPRetAux[i][j][k] = 0;		
		SinogramPtr->MagPRetDual[i][j][k] = 0;		
		SinogramPtr->PhasePRetAux[i][j][k] = 0;		
		SinogramPtr->PhasePRetDual[i][j][k] = 0;		
		
		SinogramPtr->MagTomoAux[i][j][k][0] = 1.5/2;		
		SinogramPtr->MagTomoAux[i][j][k][1] = 0.5;	
		SinogramPtr->MagTomoAux[i][j][k][2] = 1;		
		SinogramPtr->MagTomoAux[i][j][k][3] = 0.5;		
		SinogramPtr->MagTomoDual[i][j][k] = 0;	
	
		SinogramPtr->PhaseTomoAux[i][j][k][0] = 1.0/2;		
		SinogramPtr->PhaseTomoAux[i][j][k][1] = 0.5;		
		SinogramPtr->PhaseTomoAux[i][j][k][2] = 0.5;		
		SinogramPtr->PhaseTomoAux[i][j][k][3] = 1;		
		SinogramPtr->PhaseTomoDual[i][j][k] = 0;

		SinogramPtr->Omega_real[i][j][k] = 1;		
		SinogramPtr->Omega_imag[i][j][k] = 0;		
		SinogramPtr->D_real[i][j][k] = EXPECTED_COUNT_MEASUREMENT;		
		SinogramPtr->D_imag[i][j][k] = 0;		
	}
	
	if (mult_idx != 0)
	{
		size = SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t*4;
		if (read_SharedBinFile_At (MAGTOMOAUX_FILENAME, &(SinogramPtr->MagTomoAux[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
		if (read_SharedBinFile_At (PHASETOMOAUX_FILENAME, &(SinogramPtr->PhaseTomoAux[0][0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
		size = SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t;
		if (read_SharedBinFile_At (MAGPRETAUX_FILENAME, &(SinogramPtr->MagPRetAux[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
		if (read_SharedBinFile_At (PHASEPRETAUX_FILENAME, &(SinogramPtr->PhasePRetAux[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
		if (read_SharedBinFile_At (OMEGAREAL_FILENAME, &(SinogramPtr->Omega_real[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
		if (read_SharedBinFile_At (OMEGAIMAG_FILENAME, &(SinogramPtr->Omega_imag[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
	}

	SinogramPtr->delta_r = SinogramPtr->Length_R/(SinogramPtr->N_r);
	SinogramPtr->delta_t = SinogramPtr->Length_T/(SinogramPtr->N_t);
	SinogramPtr->R0 = -TomoInputsPtr->RotCenter*SinogramPtr->delta_r;
	SinogramPtr->RMax = (SinogramPtr->N_r-TomoInputsPtr->RotCenter)*SinogramPtr->delta_r;
	SinogramPtr->T0 = -SinogramPtr->Length_T/2.0;
	SinogramPtr->TMax = SinogramPtr->Length_T/2.0;
	
	/*Initializing parameters of the object to be reconstructed*/
	ScannedObjectPtr->Length_X = SinogramPtr->Length_R;
    	ScannedObjectPtr->Length_Y = SinogramPtr->Length_R;
	ScannedObjectPtr->Length_Z = SinogramPtr->Length_T;
    	ScannedObjectPtr->N_x = (int32_t)(SinogramPtr->N_r/ScannedObjectPtr->mult_xy);
	ScannedObjectPtr->N_y = (int32_t)(SinogramPtr->N_r/ScannedObjectPtr->mult_xy);
	ScannedObjectPtr->N_z = (int32_t)(SinogramPtr->N_t/ScannedObjectPtr->mult_z);	
	ScannedObjectPtr->delta_xy = ScannedObjectPtr->mult_xy*SinogramPtr->delta_r;
	ScannedObjectPtr->delta_z = ScannedObjectPtr->mult_z*SinogramPtr->delta_t;
	SinogramPtr->z_overlap_num = ScannedObjectPtr->mult_z;

	if (ScannedObjectPtr->delta_xy != ScannedObjectPtr->delta_z)
		check_warn (TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Voxel width in x-y plane is not equal to that along z-axis. The spatial invariance of prior does not hold.\n");

	ScannedObjectPtr->x0 = SinogramPtr->R0;
    	ScannedObjectPtr->z0 = SinogramPtr->T0;
    	ScannedObjectPtr->y0 = -ScannedObjectPtr->Length_Y/2.0;
    	ScannedObjectPtr->BeamWidth = SinogramPtr->delta_r; /*Weighting of the projections at different points of the detector*/

	Real_arr_t* object = (Real_arr_t*)get_spc(ScannedObjectPtr->N_time*(ScannedObjectPtr->N_z + 2)*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x, sizeof(Real_arr_t));
	ScannedObjectPtr->MagObject = Arr1DToArr4D (object, ScannedObjectPtr->N_time, ScannedObjectPtr->N_z + 2, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	object = (Real_arr_t*)get_spc(ScannedObjectPtr->N_time*(ScannedObjectPtr->N_z + 2)*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x, sizeof(Real_arr_t));
	ScannedObjectPtr->PhaseObject = Arr1DToArr4D (object, ScannedObjectPtr->N_time, ScannedObjectPtr->N_z + 2, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	
/*	ScannedObjectPtr->Object = (Real_arr_t****)get_spc(ScannedObjectPtr->N_time, sizeof(Real_arr_t***));
	for (i = 0; i < ScannedObjectPtr->N_time; i++)
	{
		ScannedObjectPtr->Object[i] = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z + 2, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	}*/
	
	/*OffsetR is stepsize of the distance between center of voxel of the object and the detector pixel, at which projections are computed*/
	SinogramPtr->OffsetR = (ScannedObjectPtr->delta_xy/sqrt(2.0)+SinogramPtr->delta_r/2.0)/DETECTOR_RESPONSE_BINS;
	SinogramPtr->OffsetT = ((ScannedObjectPtr->delta_z/2) + SinogramPtr->delta_t/2)/DETECTOR_RESPONSE_BINS;

	TomoInputsPtr->num_threads = omp_get_max_threads();
	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Maximum number of openmp threads is %d\n", TomoInputsPtr->num_threads);
	if (TomoInputsPtr->num_threads <= 1)
		check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "The maximum number of threads is less than or equal to 1.\n");	
	TomoInputsPtr->num_z_blocks = TomoInputsPtr->num_threads/ScannedObjectPtr->N_time;
	if (TomoInputsPtr->num_z_blocks < 2)
		TomoInputsPtr->num_z_blocks = 2;
	else if (TomoInputsPtr->num_z_blocks > ScannedObjectPtr->N_z)
		TomoInputsPtr->num_z_blocks = ScannedObjectPtr->N_z;
	TomoInputsPtr->num_z_blocks = (TomoInputsPtr->num_z_blocks/2)*2; /*Round down to the nearest even integer*/
	
	TomoInputsPtr->prevnum_z_blocks = TomoInputsPtr->num_threads/ScannedObjectPtr->N_time;
	if (TomoInputsPtr->prevnum_z_blocks < 2)
		TomoInputsPtr->prevnum_z_blocks = 2;
	else 
	{
		if (TomoInputsPtr->initICD == 3 && TomoInputsPtr->prevnum_z_blocks > ScannedObjectPtr->N_z/2)
			TomoInputsPtr->prevnum_z_blocks = ScannedObjectPtr->N_z/2;
		else if (TomoInputsPtr->prevnum_z_blocks > ScannedObjectPtr->N_z)
			TomoInputsPtr->prevnum_z_blocks = ScannedObjectPtr->N_z;
	}
	TomoInputsPtr->prevnum_z_blocks = (TomoInputsPtr->prevnum_z_blocks/2)*2; /*Round down to the nearest even integer*/

/*	TomoInputsPtr->BoundaryFlag = (uint8_t***)multialloc(sizeof(uint8_t), 3, 3, 3, 3);*/
        TomoInputsPtr->x_rand_select = (int32_t***)multialloc(sizeof(int32_t), 3, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->y_rand_select = (int32_t***)multialloc(sizeof(int32_t), 3, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->x_NHICD_select = (int32_t***)multialloc(sizeof(int32_t), 3, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->y_NHICD_select = (int32_t***)multialloc(sizeof(int32_t), 3, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->UpdateSelectNum = (int32_t**)multialloc(sizeof(int32_t), 2, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks);
        TomoInputsPtr->NHICDSelectNum = (int32_t**)multialloc(sizeof(int32_t), 2, ScannedObjectPtr->N_time, TomoInputsPtr->num_z_blocks);

	ScannedObjectPtr->NHICD_Iterations = 10;

	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of z blocks is %d\n", TomoInputsPtr->num_z_blocks);
	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of z blocks in previous multi-resolution stage is %d\n", TomoInputsPtr->prevnum_z_blocks);

	SinogramPtr->ViewPtr = (Real_arr_t*)get_spc(proj_num, sizeof(Real_arr_t));
	SinogramPtr->TimePtr = (Real_arr_t*)get_spc(proj_num, sizeof(Real_arr_t));
	for (i = 0; i < proj_num; i++)
	{
		SinogramPtr->ViewPtr[i] = proj_angles[i];
		SinogramPtr->TimePtr[i] = proj_times[i];
	}

	ScannedObjectPtr->recon_times = (Real_arr_t*)get_spc(recon_num+1, sizeof(Real_arr_t));
	for (i = 0; i < recon_num+1; i++)
		ScannedObjectPtr->recon_times[i] = recon_times[i];
	flag = initRandomAngles (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
	/*TomoInputs holds the input parameters and some miscellaneous variables*/
	TomoInputsPtr->Mag_Sigma_S_Q = pow((ScannedObjectPtr->Mag_Sigma_S*ScannedObjectPtr->mult_xy),MRF_Q);
	TomoInputsPtr->Mag_Sigma_S_Q_P = pow(ScannedObjectPtr->Mag_Sigma_S*ScannedObjectPtr->mult_xy,MRF_Q-MRF_P);	
	TomoInputsPtr->Mag_Sigma_T_Q = pow((ScannedObjectPtr->Mag_Sigma_T*ScannedObjectPtr->delta_recon),MRF_Q);
	TomoInputsPtr->Mag_Sigma_T_Q_P = pow(ScannedObjectPtr->Mag_Sigma_T*ScannedObjectPtr->delta_recon,MRF_Q-MRF_P);
	
	TomoInputsPtr->Phase_Sigma_S_Q = pow((ScannedObjectPtr->Phase_Sigma_S*ScannedObjectPtr->mult_xy),MRF_Q);
	TomoInputsPtr->Phase_Sigma_S_Q_P = pow(ScannedObjectPtr->Phase_Sigma_S*ScannedObjectPtr->mult_xy,MRF_Q-MRF_P);	
	TomoInputsPtr->Phase_Sigma_T_Q = pow((ScannedObjectPtr->Phase_Sigma_T*ScannedObjectPtr->delta_recon),MRF_Q);
	TomoInputsPtr->Phase_Sigma_T_Q_P = pow(ScannedObjectPtr->Phase_Sigma_T*ScannedObjectPtr->delta_recon,MRF_Q-MRF_P);
	initFilter (ScannedObjectPtr, TomoInputsPtr);
	
	calculateSinCos (SinogramPtr, TomoInputsPtr);
	TomoInputsPtr->ADMM_mu = 1;	
	TomoInputsPtr->NMS_rho = 1;	
	TomoInputsPtr->NMS_chi = 2;	
	TomoInputsPtr->NMS_gamma = 0.5;	
	TomoInputsPtr->NMS_sigma = 0.5;	
	TomoInputsPtr->NMS_threshold = 0.05;
	TomoInputsPtr->NMS_MaxIter = 50;
	TomoInputsPtr->MaxHeadIter = 20;
	TomoInputsPtr->PRetMaxIter = 10;
	TomoInputsPtr->SteepDesMaxIter = 50;

	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Initialized the structures, Sinogram and ScannedObject\n");
	
	check_error(SinogramPtr->N_t % (int32_t)ScannedObjectPtr->mult_z != 0, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cannot do reconstruction since mult_z = %d does not divide %d\n", (int32_t)ScannedObjectPtr->mult_z, SinogramPtr->N_t);
	check_error(SinogramPtr->N_r % (int32_t)ScannedObjectPtr->mult_xy != 0, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Cannot do reconstruction since mult_xy = %d does not divide %d\n", (int32_t)ScannedObjectPtr->mult_xy, SinogramPtr->N_r);
	return (flag);
error:
	return (-1);	
}


/*Free memory of several arrays*/
void freeMemory(Sinogram* SinogramPtr, ScannedObject *ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	int32_t i;
	for (i=0; i<ScannedObjectPtr->N_time; i++)
		{if (ScannedObjectPtr->ProjIdxPtr[i]) free(ScannedObjectPtr->ProjIdxPtr[i]);}
	
	if (ScannedObjectPtr->ProjIdxPtr) free(ScannedObjectPtr->ProjIdxPtr);
	if (ScannedObjectPtr->ProjNum) free(ScannedObjectPtr->ProjNum);
	if (SinogramPtr->ViewPtr) free(SinogramPtr->ViewPtr);
	if (SinogramPtr->TimePtr) free(SinogramPtr->TimePtr);
	if (ScannedObjectPtr->recon_times) free(ScannedObjectPtr->recon_times);

/*	for (i = 0; i < ScannedObjectPtr->N_time; i++)
		{if (ScannedObjectPtr->Object[i]) multifree(ScannedObjectPtr->Object[i],3);}
	if (ScannedObjectPtr->Object) free(ScannedObjectPtr->Object);*/
/*	multifree(ScannedObjectPtr->Object, 4);*/
	
	if (SinogramPtr->Measurements) multifree(SinogramPtr->Measurements,3);
	if (SinogramPtr->MagTomoAux) multifree(SinogramPtr->MagTomoAux,4);
	if (SinogramPtr->MagTomoDual) multifree(SinogramPtr->MagTomoDual,3);
	if (SinogramPtr->PhaseTomoAux) multifree(SinogramPtr->PhaseTomoAux,4);
	if (SinogramPtr->PhaseTomoDual) multifree(SinogramPtr->PhaseTomoDual,3);
	
	if (SinogramPtr->MagPRetAux) multifree(SinogramPtr->MagPRetAux,3);
	if (SinogramPtr->MagPRetDual) multifree(SinogramPtr->MagPRetDual,3);
	if (SinogramPtr->PhasePRetAux) multifree(SinogramPtr->PhasePRetAux,3);
	if (SinogramPtr->PhasePRetDual) multifree(SinogramPtr->PhasePRetDual,3);
	
	if (SinogramPtr->Omega_real) multifree(SinogramPtr->Omega_real,3);
	if (SinogramPtr->Omega_imag) multifree(SinogramPtr->Omega_imag,3);
	if (SinogramPtr->D_real) multifree(SinogramPtr->D_real,3);
	if (SinogramPtr->D_imag) multifree(SinogramPtr->D_imag,3);

	if (TomoInputsPtr->x_rand_select) multifree(TomoInputsPtr->x_rand_select,3);
	if (TomoInputsPtr->y_rand_select) multifree(TomoInputsPtr->y_rand_select,3);
	if (TomoInputsPtr->x_NHICD_select) multifree(TomoInputsPtr->x_NHICD_select,3);
	if (TomoInputsPtr->y_NHICD_select) multifree(TomoInputsPtr->y_NHICD_select,3);
	if (TomoInputsPtr->UpdateSelectNum) multifree(TomoInputsPtr->UpdateSelectNum,2);
	if (TomoInputsPtr->NHICDSelectNum) multifree(TomoInputsPtr->NHICDSelectNum,2);
	if (TomoInputsPtr->Weight) multifree(TomoInputsPtr->Weight,3);	
	if (SinogramPtr->cosine) free(SinogramPtr->cosine);
	if (SinogramPtr->sine) free(SinogramPtr->sine);
}

int32_t initPhantomStructures (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, float* projections, float* weights, float *proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, Real_t vox_wid, Real_t rot_center)
{
	int i;
	MPI_Comm_size(MPI_COMM_WORLD, &(TomoInputsPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(TomoInputsPtr->node_rank));
	
	SinogramPtr->Length_R = vox_wid*proj_cols;
	SinogramPtr->Length_T = vox_wid*proj_rows;
	TomoInputsPtr->RotCenter = rot_center;
		
	TomoInputsPtr->Write2Tiff = ENABLE_TIFF_WRITES;
	SinogramPtr->N_p = proj_num;	
	SinogramPtr->N_r = proj_cols;
	SinogramPtr->total_t_slices = proj_rows;

	SinogramPtr->Length_T = SinogramPtr->Length_T/TomoInputsPtr->node_num;
	SinogramPtr->N_t = SinogramPtr->total_t_slices/TomoInputsPtr->node_num;
	
	SinogramPtr->delta_r = SinogramPtr->Length_R/(SinogramPtr->N_r);
	SinogramPtr->delta_t = SinogramPtr->Length_T/(SinogramPtr->N_t);
	SinogramPtr->R0 = -TomoInputsPtr->RotCenter*SinogramPtr->delta_r;
	SinogramPtr->RMax = (SinogramPtr->N_r-TomoInputsPtr->RotCenter)*SinogramPtr->delta_r;
	SinogramPtr->T0 = -SinogramPtr->Length_T/2.0;
	SinogramPtr->TMax = SinogramPtr->Length_T/2.0;
	
	/*Initializing parameters of the object to be reconstructed*/
	ScannedObjectPtr->Length_X = SinogramPtr->Length_R;
    	ScannedObjectPtr->Length_Y = SinogramPtr->Length_R;
	ScannedObjectPtr->Length_Z = SinogramPtr->Length_T;
	
	check_error (PHANTOM_XY_SIZE % SinogramPtr->N_r != 0 || PHANTOM_Z_SIZE % SinogramPtr->total_t_slices != 0, TomoInputsPtr->node_rank == 0, TomoInputsPtr->debug_file_ptr, "proj_cols = %d does not divide PHANTOM_XY_SIZE = %d or proj_rows = %d does not divide PHANTOM_Z_SIZE = %d\n", SinogramPtr->N_r, PHANTOM_XY_SIZE, SinogramPtr->N_t, PHANTOM_Z_SIZE);
	
	int32_t phantom_mult_xy = PHANTOM_XY_SIZE/SinogramPtr->N_r;
	int32_t phantom_mult_z = PHANTOM_Z_SIZE/SinogramPtr->total_t_slices;
    	ScannedObjectPtr->N_x = (int32_t)(PHANTOM_XY_SIZE);
	ScannedObjectPtr->N_y = (int32_t)(PHANTOM_XY_SIZE);
	ScannedObjectPtr->N_z = (int32_t)(PHANTOM_Z_SIZE);	
	ScannedObjectPtr->delta_xy = SinogramPtr->delta_r/phantom_mult_xy;
	ScannedObjectPtr->delta_z = SinogramPtr->delta_t/phantom_mult_z;

	if (ScannedObjectPtr->delta_xy != ScannedObjectPtr->delta_z)
		check_warn (TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Voxel width in x-y plane is not equal to that along z-axis. The spatial invariance of prior does not hold.\n");

	ScannedObjectPtr->x0 = SinogramPtr->R0;
    	ScannedObjectPtr->z0 = SinogramPtr->T0;
    	ScannedObjectPtr->y0 = -ScannedObjectPtr->Length_Y/2.0;
    	ScannedObjectPtr->BeamWidth = SinogramPtr->delta_r; /*Weighting of the projections at different points of the detector*/
/*	ScannedObjectPtr->Object = (Real_t****)multialloc(sizeof(Real_t), 4, ScannedObjectPtr->N_time, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, ScannedObjectPtr->N_z);*/
	/*OffsetR is stepsize of the distance between center of voxel of the object and the detector pixel, at which projections are computed*/
	SinogramPtr->OffsetR = (ScannedObjectPtr->delta_xy/sqrt(2.0) + SinogramPtr->delta_r/2.0)/DETECTOR_RESPONSE_BINS;
	SinogramPtr->OffsetT = ((ScannedObjectPtr->delta_z/2) + SinogramPtr->delta_t/2)/DETECTOR_RESPONSE_BINS;

	TomoInputsPtr->num_threads = omp_get_max_threads();
	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Maximum number of openmp threads is %d\n", TomoInputsPtr->num_threads);
	if (TomoInputsPtr->num_threads <= 1)
		check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "The maximum number of threads is less than or equal to 1.\n");	
	
	SinogramPtr->ViewPtr = (Real_arr_t*)get_spc(proj_num, sizeof(Real_arr_t));
	for (i = 0; i < proj_num; i++)
		SinogramPtr->ViewPtr[i] = proj_angles[i];
	calculateSinCos (SinogramPtr, TomoInputsPtr);
	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Initialized the structures, Sinogram and ScannedObject\n");
	
	return (0);
error:
	return (-1);
}

/*Free memory of several arrays*/
void freePhantomMemory(Sinogram* SinogramPtr, ScannedObject *ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
/*	int32_t i;
	for (i=0; i<ScannedObjectPtr->N_time; i++)
		{if (ScannedObjectPtr->ProjIdxPtr[i]) free(ScannedObjectPtr->ProjIdxPtr[i]);}
	
	if (ScannedObjectPtr->ProjIdxPtr) free(ScannedObjectPtr->ProjIdxPtr);
	if (ScannedObjectPtr->ProjNum) free(ScannedObjectPtr->ProjNum);
*/
/*	for (i = 0; i < ScannedObjectPtr->N_time; i++)
		{if (ScannedObjectPtr->Object[i]) multifree(ScannedObjectPtr->Object[i],3);}
	if (ScannedObjectPtr->Object) free(ScannedObjectPtr->Object);*/
/*	multifree(ScannedObjectPtr->Object, 4);*/
	
/*	if (SinogramPtr->Measurements) multifree(SinogramPtr->Measurements,3);
	if (SinogramPtr->MagTomoAux) multifree(SinogramPtr->MagTomoAux,3);
	if (SinogramPtr->MagTomoDual) multifree(SinogramPtr->MagTomoDual,3);
	if (SinogramPtr->PhaseTomoAux) multifree(SinogramPtr->PhaseTomoAux,3);
	if (SinogramPtr->PhaseTomoDual) multifree(SinogramPtr->PhaseTomoDual,3);
*/
/*	if (TomoInputsPtr->x_rand_select) multifree(TomoInputsPtr->x_rand_select,3);
	if (TomoInputsPtr->y_rand_select) multifree(TomoInputsPtr->y_rand_select,3);
	if (TomoInputsPtr->x_NHICD_select) multifree(TomoInputsPtr->x_NHICD_select,3);
	if (TomoInputsPtr->y_NHICD_select) multifree(TomoInputsPtr->y_NHICD_select,3);
	if (TomoInputsPtr->UpdateSelectNum) multifree(TomoInputsPtr->UpdateSelectNum,2);
	if (TomoInputsPtr->NHICDSelectNum) multifree(TomoInputsPtr->NHICDSelectNum,2);
	if (TomoInputsPtr->Weight) multifree(TomoInputsPtr->Weight,3);	*/
	if (SinogramPtr->ViewPtr) free(SinogramPtr->ViewPtr);
	if (SinogramPtr->cosine) free(SinogramPtr->cosine);
	if (SinogramPtr->sine) free(SinogramPtr->sine);
}


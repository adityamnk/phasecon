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
#include "XT_PhaseRet.h"


/*Computes the Euclidean distance of a voxel to its neighboring voxels
--Inputs--
i, j, k, l are the indices of neighoring voxels
--Outputs--
Returns the distance */
Real_t distance2node(uint8_t i, uint8_t j, uint8_t k)
{
	return(sqrt(pow((Real_t)i-1.0, 2.0)+pow((Real_t)j-1.0, 2.0)+pow((Real_t)k-1.0, 2.0)));
}

/*Initializes the weights w_{ij} used in the qGGMRF models*/
void initFilter (ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	uint8_t i,j,k;
	Real_t temp1,sum=0,prior_const=0;
/*	prior_const = ScannedObjectPtr->delta_xy*ScannedObjectPtr->delta_xy*ScannedObjectPtr->delta_xy*ScannedObjectPtr->delta_Rtime;*/
	prior_const = ScannedObjectPtr->mult_xy*ScannedObjectPtr->mult_xy*ScannedObjectPtr->mult_xy;
/*Filter coefficients of neighboring pixels are inversely proportional to the distance from the center pixel*/
	
	for (i=0; i<3; i++)
	for (j=0; j<3; j++)
	for (k=0; k<3; k++){
	if(i!=1 || j!=1 || k!=1)
	{
		temp1 = 1.0/distance2node(i,j,k);
		TomoInputsPtr->Spatial_Filter[i][j][k] = temp1;
		sum=sum+temp1;
	}
	else
		TomoInputsPtr->Spatial_Filter[i][j][k]=0;
	}

	for (i=0; i<3; i++)
	for (j=0; j<3; j++)
	for (k=0; k<3; k++){
		TomoInputsPtr->Spatial_Filter[i][j][k] = prior_const*TomoInputsPtr->Spatial_Filter[i][j][k]/sum;
	}


#ifdef EXTRA_DEBUG_MESSAGES
	sum=0;
	for (i=0; i<3; i++)
		for (j=0; j<3; j++)
			for (k=0; k<3; k++)
			{
				sum += TomoInputsPtr->Spatial_Filter[i][j][k];
				check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "initFilter: Filter i=%d, j=%d, k=%d, coeff = %f\n", i,j,k,TomoInputsPtr->Spatial_Filter[i][j][k]/prior_const);
			}
			check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "initFilter: Sum of filter coefficients is %f\n",(sum)/prior_const);	
			check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "initFilter: delta_xy*delta_xy*delta_z = %f\n",prior_const);	
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
int32_t initStructures (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, int32_t mult_idx, int32_t mult_xy[], int32_t mult_z[], float *data_unflip_x, float *data_flip_x, float *data_unflip_y, float *data_flip_y, float *proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, Real_t vox_wid, Real_t rot_center, Real_t mag_sigma, Real_t mag_c, Real_t elec_sigma, Real_t elec_c, Real_t convg_thresh)
{
	int flag = 0, i;

	/*MPI node number and total node count parameters*/
	MPI_Comm_size(MPI_COMM_WORLD, &(TomoInputsPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(TomoInputsPtr->node_rank));

	ScannedObjectPtr->Mag_Sigma[0] = mag_sigma;
	ScannedObjectPtr->Mag_Sigma[1] = mag_sigma;
	ScannedObjectPtr->Mag_Sigma[2] = mag_sigma;
	ScannedObjectPtr->Mag_C[0] = mag_c;
	ScannedObjectPtr->Mag_C[1] = mag_c;
	ScannedObjectPtr->Mag_C[2] = mag_c;
	
	ScannedObjectPtr->Elec_Sigma = elec_sigma;
	ScannedObjectPtr->Elec_C = elec_c;

	TomoInputsPtr->Weight = 1;
	
	ScannedObjectPtr->mult_xy = mult_xy[mult_idx];
	ScannedObjectPtr->mult_z = mult_z[mult_idx];		
	SinogramPtr->Length_R = vox_wid*proj_cols;
	SinogramPtr->Length_T = vox_wid*proj_rows;
	TomoInputsPtr->StopThreshold = convg_thresh;
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
	SinogramPtr->N_p = proj_num;	
	SinogramPtr->N_r = proj_cols;
	TomoInputsPtr->cost_thresh = COST_CONVG_THRESHOLD;	
	TomoInputsPtr->radius_obj = vox_wid*proj_cols;	
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
	
	SinogramPtr->Data_Unflip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->Data_Flip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

#ifdef VFET_TWO_AXES
	SinogramPtr->Data_Unflip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);
	SinogramPtr->Data_Flip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_r, SinogramPtr->N_t);

	for (i = 0; i < SinogramPtr->N_p; i++)
	for (j = 0; j < SinogramPtr->N_r; j++)
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		idx = i*SinogramPtr->N_t*SinogramPtr->N_r + k*SinogramPtr->N_r + j;
		SinogramPtr->Data_Unflip_y[i][j][k] = data_unflip_y[idx];
		SinogramPtr->Data_Flip_y[i][j][k] = data_flip_y[idx];
	}
#endif
	
	for (i = 0; i < SinogramPtr->N_p; i++)
	for (j = 0; j < SinogramPtr->N_r; j++)
	for (k = 0; k < SinogramPtr->N_t; k++)
	{
		idx = i*SinogramPtr->N_t*SinogramPtr->N_r + j*SinogramPtr->N_t + k;
		SinogramPtr->Data_Unflip_x[i][j][k] = data_unflip_x[idx];
		SinogramPtr->Data_Flip_x[i][j][k] = data_flip_x[idx];
	}

	int dimTiff[4];
    	dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_r; dimTiff[3] = SinogramPtr->N_t;
    	if (TomoInputsPtr->Write2Tiff == 1)
	{
    		if (WriteMultiDimArray2Tiff (DATA_UNFLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->Data_Unflip_x[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    		if (WriteMultiDimArray2Tiff (DATA_FLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->Data_Flip_x[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
#ifdef VFET_TWO_AXES
    		if (WriteMultiDimArray2Tiff (DATA_UNFLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->Data_Unflip_y[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
    		if (WriteMultiDimArray2Tiff (DATA_FLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinogramPtr->Data_Flip_y[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) goto error;
#endif
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

	ScannedObjectPtr->MagPotentials = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ScannedObjectPtr->N_z + 2, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, 3);
	ScannedObjectPtr->ElecPotentials = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z + 2, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);

	/*OffsetR is stepsize of the distance between center of voxel of the object and the detector pixel, at which projections are computed*/
	SinogramPtr->OffsetR = (ScannedObjectPtr->delta_xy/sqrt(2.0)+SinogramPtr->delta_r/2.0)/DETECTOR_RESPONSE_BINS;
	SinogramPtr->OffsetT = ((ScannedObjectPtr->delta_z/2) + SinogramPtr->delta_t/2)/DETECTOR_RESPONSE_BINS;

	TomoInputsPtr->num_threads = omp_get_max_threads();
	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Maximum number of openmp threads is %d\n", TomoInputsPtr->num_threads);
	if (TomoInputsPtr->num_threads <= 1)
		check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "The maximum number of threads is less than or equal to 1.\n");	
	TomoInputsPtr->num_z_blocks = TomoInputsPtr->num_threads;
	if (TomoInputsPtr->num_z_blocks < 2)
		TomoInputsPtr->num_z_blocks = 2;
	else if (TomoInputsPtr->num_z_blocks > ScannedObjectPtr->N_z)
		TomoInputsPtr->num_z_blocks = ScannedObjectPtr->N_z;
	TomoInputsPtr->num_z_blocks = (TomoInputsPtr->num_z_blocks/2)*2; /*Round down to the nearest even integer*/
	
	TomoInputsPtr->prevnum_z_blocks = TomoInputsPtr->num_threads;
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
        TomoInputsPtr->x_rand_select = (int32_t**)multialloc(sizeof(int32_t), 2, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->y_rand_select = (int32_t**)multialloc(sizeof(int32_t), 2, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->x_NHICD_select = (int32_t**)multialloc(sizeof(int32_t), 2, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->y_NHICD_select = (int32_t**)multialloc(sizeof(int32_t), 2, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y*ScannedObjectPtr->N_x);
        TomoInputsPtr->UpdateSelectNum = (int32_t*)multialloc(sizeof(int32_t), 1, TomoInputsPtr->num_z_blocks);
        TomoInputsPtr->NHICDSelectNum = (int32_t*)multialloc(sizeof(int32_t), 1, TomoInputsPtr->num_z_blocks);

	ScannedObjectPtr->NHICD_Iterations = 10;

	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of z blocks is %d\n", TomoInputsPtr->num_z_blocks);
	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of z blocks in previous multi-resolution stage is %d\n", TomoInputsPtr->prevnum_z_blocks);

	SinogramPtr->ViewPtr = (Real_arr_t*)get_spc(proj_num, sizeof(Real_arr_t));
	for (i = 0; i < proj_num; i++)
		SinogramPtr->ViewPtr[i] = proj_angles[i];

	/*TomoInputs holds the input parameters and some miscellaneous variables*/
	for (i = 0; i < 3; i++)
	{
		TomoInputsPtr->Mag_Sigma_Q[i] = pow((ScannedObjectPtr->Mag_Sigma[i]*ScannedObjectPtr->mult_xy),MRF_Q);
		TomoInputsPtr->Mag_Sigma_Q_P[i] = pow(ScannedObjectPtr->Mag_Sigma[i]*ScannedObjectPtr->mult_xy,MRF_Q-MRF_P);	
	}

	TomoInputsPtr->Elec_Sigma_Q = pow((ScannedObjectPtr->Elec_Sigma*ScannedObjectPtr->mult_xy),MRF_Q);
	TomoInputsPtr->Elec_Sigma_Q_P = pow(ScannedObjectPtr->Elec_Sigma*ScannedObjectPtr->mult_xy,MRF_Q-MRF_P);	
	initFilter (ScannedObjectPtr, TomoInputsPtr);
	
	calculateSinCos (SinogramPtr, TomoInputsPtr);
	
	TomoInputsPtr->ADMM_mu = 1;	

	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "The ADMM mu is %f.\n", TomoInputsPtr->ADMM_mu);
		
	TomoInputsPtr->NumIter = MAX_NUM_ITERATIONS;
	TomoInputsPtr->Head_MaxIter = 100;
	
	TomoInputsPtr->Head_threshold = 0.001;

#ifdef INIT_GROUND_TRUTH_PHANTOM
	ScannedObjectPtr->ElecPotGndTruth = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, PHANTOM_Z_SIZE, PHANTOM_Y_SIZE, PHANTOM_X_SIZE);
	ScannedObjectPtr->MagPotGndTruth = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, PHANTOM_Z_SIZE, PHANTOM_Y_SIZE, PHANTOM_X_SIZE, 3);
	size = PHANTOM_Z_SIZE*PHANTOM_Y_SIZE*PHANTOM_X_SIZE;
	if (read_SharedBinFile_At (PHANTOM_ELECOBJECT_FILENAME, &(ScannedObjectPtr->ElecPotGndTruth[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
	if (read_SharedBinFile_At (PHANTOM_MAGOBJECT_FILENAME, &(ScannedObjectPtr->MagPotGndTruth[0][0][0][0]), TomoInputsPtr->node_rank*size*3, size*3, TomoInputsPtr->debug_file_ptr)) flag = -1;
#endif
		

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
#ifdef INIT_GROUND_TRUTH_PHANTOM
	multifree(ScannedObjectPtr->MagPotGndTruth,4);
	multifree(ScannedObjectPtr->ElecPotGndTruth,3);
#endif
	multifree(ScannedObjectPtr->MagPotentials,4);
	multifree(ScannedObjectPtr->ElecPotentials,3);
	if (SinogramPtr->ViewPtr) free(SinogramPtr->ViewPtr);

	if (SinogramPtr->Data_Unflip_x) multifree(SinogramPtr->Data_Unflip_x,3);
	if (SinogramPtr->Data_Flip_x) multifree(SinogramPtr->Data_Flip_x,3);
#ifdef VFET_TWO_AXES
	if (SinogramPtr->Data_Unflip_y) multifree(SinogramPtr->Data_Unflip_y,3);
	if (SinogramPtr->Data_Flip_y) multifree(SinogramPtr->Data_Flip_y,3);
#endif
	
	if (TomoInputsPtr->x_rand_select) multifree(TomoInputsPtr->x_rand_select,2);
	if (TomoInputsPtr->y_rand_select) multifree(TomoInputsPtr->y_rand_select,2);
	if (TomoInputsPtr->x_NHICD_select) multifree(TomoInputsPtr->x_NHICD_select,2);
	if (TomoInputsPtr->y_NHICD_select) multifree(TomoInputsPtr->y_NHICD_select,2);
	if (TomoInputsPtr->UpdateSelectNum) multifree(TomoInputsPtr->UpdateSelectNum,1);
	if (TomoInputsPtr->NHICDSelectNum) multifree(TomoInputsPtr->NHICDSelectNum,1);
	if (SinogramPtr->cosine) free(SinogramPtr->cosine);
	if (SinogramPtr->sine) free(SinogramPtr->sine);
}


int32_t initPhantomStructures (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, float *proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, Real_t vox_wid, Real_t rot_center)
{
	int i;
	
	MPI_Comm_size(MPI_COMM_WORLD, &(TomoInputsPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(TomoInputsPtr->node_rank));
	
	SinogramPtr->Length_R = vox_wid*proj_cols;
	SinogramPtr->Length_T = vox_wid*proj_rows;
	TomoInputsPtr->RotCenter = rot_center*(PHANTOM_XY_SIZE/proj_cols);
		
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
	
    	ScannedObjectPtr->N_x = (int32_t)(PHANTOM_XY_SIZE);
	ScannedObjectPtr->N_y = (int32_t)(PHANTOM_XY_SIZE);
	ScannedObjectPtr->N_z = (int32_t)(PHANTOM_Z_SIZE);	
	ScannedObjectPtr->delta_xy = SinogramPtr->delta_r;
	ScannedObjectPtr->delta_z = SinogramPtr->delta_t;

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
	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Projection angles are - ");
	for (i = 0; i < proj_num; i++)
	{
		SinogramPtr->ViewPtr[i] = proj_angles[i];
		check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "(%d,%f)", i, SinogramPtr->ViewPtr[i]);
	}
	check_info(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "\n");

	calculateSinCos (SinogramPtr, TomoInputsPtr);
	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Initialized the structures, Sinogram and ScannedObject\n");
	
	return (0);
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


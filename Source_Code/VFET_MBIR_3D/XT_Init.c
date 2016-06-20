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
/*#include "XT_MPI.h"
#include "XT_MPIIO.h"*/
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
void initFilter (ScannedObject* ObjPtr, TomoInputs* InpPtr)
{
	uint8_t i,j,k;
	Real_t temp1,sum=0,prior_const=0;
/*	prior_const = ObjPtr->delta_xy*ObjPtr->delta_xy*ObjPtr->delta_xy*ObjPtr->delta_Rtime;*/
	prior_const = ObjPtr->mult_xy*ObjPtr->mult_xy*ObjPtr->mult_xy;
/*Filter coefficients of neighboring pixels are inversely proportional to the distance from the center pixel*/
	
	for (i=0; i<3; i++)
	for (j=0; j<3; j++)
	for (k=0; k<3; k++){
	if(i!=1 || j!=1 || k!=1)
	{
		temp1 = 1.0/distance2node(i,j,k);
		InpPtr->Spatial_Filter[i][j][k] = temp1;
		sum=sum+temp1;
	}
	else
		InpPtr->Spatial_Filter[i][j][k]=0;
	}

	for (i=0; i<3; i++)
	for (j=0; j<3; j++)
	for (k=0; k<3; k++){
		InpPtr->Spatial_Filter[i][j][k] = prior_const*InpPtr->Spatial_Filter[i][j][k]/sum;
	}


#ifdef EXTRA_DEBUG_MESSAGES
	sum=0;
	for (i=0; i<3; i++)
		for (j=0; j<3; j++)
			for (k=0; k<3; k++)
			{
				sum += InpPtr->Spatial_Filter[i][j][k];
				check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "initFilter: Filter i=%d, j=%d, k=%d, coeff = %f\n", i,j,k,InpPtr->Spatial_Filter[i][j][k]/prior_const);
			}
			check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "initFilter: Sum of filter coefficients is %f\n",(sum)/prior_const);	
			check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "initFilter: delta_xy*delta_xy*delta_z = %f\n",prior_const);	
#endif /*#ifdef DEBUG_EN*/


}

/*Initializes the sines and cosines of angles at which projections are acquired. It is then used when computing the voxel profile*/
void calculateSinCos(Sinogram* SinoPtr, TomoInputs* InpPtr)
{
  int32_t i;

  SinoPtr->cosine=(Real_t*)get_spc(SinoPtr->N_p, sizeof(Real_t));
  SinoPtr->sine=(Real_t*)get_spc(SinoPtr->N_p, sizeof(Real_t));

  for(i=0;i<SinoPtr->N_p;i++)
  {
    SinoPtr->cosine[i]=cos(SinoPtr->ViewPtr[i]);
    SinoPtr->sine[i]=sin(SinoPtr->ViewPtr[i]);
  }
  check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "calculateSinCos: Calculated sines and cosines of angles of rotation\n");
}


void initCrossProdFilter (ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftarr)
{
	int32_t i, j, k, idx_i, idx_j, idx_k, idx;
	Real_t dist, delta, distmag, distelec, h0_mag, h0_elec, magxr, magyr, magzr, elecr, magxi, magyi, magzi, eleci;

	delta = ObjPtr->delta_xy/(2.0*CROSSPROD_IMP_WIDTH+1);
	h0_mag = 0; h0_elec = 0;
	for (i = -CROSSPROD_IMP_WIDTH; i <= CROSSPROD_IMP_WIDTH; i++)
	for (j = -CROSSPROD_IMP_WIDTH; j <= CROSSPROD_IMP_WIDTH; j++)
	for (k = -CROSSPROD_IMP_WIDTH; k <= CROSSPROD_IMP_WIDTH; k++)
	{
		distmag = pow(sqrt((Real_t)(i*i + j*j + k*k)), 3);
		distelec = sqrt((Real_t)(i*i + j*j + k*k));
		if (i != 0 || j != 0 || k != 0)
		{
			h0_mag += ((Real_t)i)/distmag;
			h0_elec += 1.0/distelec;
		}
		else
		{
			h0_mag += 0;
			h0_elec += 1.0;
		}
	}
	h0_mag *= delta;
	h0_elec *= delta*delta;

	fprintf(InpPtr->debug_file_ptr, "initCrossProdFilter: Zero'th impulse function value is (mag,elec) = (%e,%e)\n", h0_mag, h0_elec);

	magxr = 0; magyr = 0; magzr = 0; elecr = 0;
	magxi = 0; magyi = 0; magzi = 0; eleci = 0;
	for (i = ObjPtr->N_z/2 - 1; i >= -ObjPtr->N_z/2 + 1; i--)
	for (j = ObjPtr->N_y/2 - 1; j >= -ObjPtr->N_y/2 + 1; j--)
	for (k = ObjPtr->N_x/2 - 1; k >= -ObjPtr->N_x/2 + 1; k--)
	{
		idx_i = (i + fftarr->z_num) % (fftarr->z_num);
		idx_j = (j + fftarr->y_num) % (fftarr->y_num);
		idx_k = (k + fftarr->x_num) % (fftarr->x_num);

		idx = idx_i*fftarr->y_num*fftarr->x_num	+ idx_j*fftarr->x_num + idx_k;	
	
		if (i != 0 || j != 0 || k != 0)
		{
			dist = pow(sqrt((Real_t)(i*i + j*j + k*k)), 3);
			fftarr->fftforw_magarr[0][idx][0] = ObjPtr->delta_xy*((Real_t)i)/dist;
			fftarr->fftforw_magarr[1][idx][0] = ObjPtr->delta_xy*((Real_t)j)/dist;
			fftarr->fftforw_magarr[2][idx][0] = ObjPtr->delta_xy*((Real_t)k)/dist;
		
			dist = sqrt((Real_t)(i*i + j*j + k*k));
			fftarr->fftforw_elecarr[idx][0] = ObjPtr->delta_xy*ObjPtr->delta_xy/dist;
		}
		else
		{
			fftarr->fftforw_magarr[0][idx][0] = 0;
			fftarr->fftforw_magarr[1][idx][0] = 0;
			fftarr->fftforw_magarr[2][idx][0] = 0;
			fftarr->fftforw_elecarr[idx][0] = h0_elec;
		}
		
		fftarr->fftforw_magarr[0][idx][1] = 0;
		fftarr->fftforw_magarr[1][idx][1] = 0;
		fftarr->fftforw_magarr[2][idx][1] = 0;
		fftarr->fftforw_elecarr[idx][1] = 0;
		
/*		printf("Space i = %d, j = %d, k = %d, idx_i = %d, idx_j = %d, idx_k = %d, idx = %d, fft (mag,elec) = ((%e,%e),(%e,%e),(%e,%e),(%e,%e))\n", i, j, k, idx_i, idx_j, idx_k, idx, fftarr->fftforw_magarr[0][idx][0], fftarr->fftforw_magarr[0][idx][1], fftarr->fftforw_magarr[1][idx][0], fftarr->fftforw_magarr[1][idx][1], fftarr->fftforw_magarr[2][idx][0], fftarr->fftforw_magarr[2][idx][1], fftarr->fftforw_elecarr[idx][0], fftarr->fftforw_elecarr[idx][1]);*/
		
		magzr += fabs(fftarr->fftforw_magarr[0][idx][0]);	
		magyr += fabs(fftarr->fftforw_magarr[1][idx][0]);	
		magxr += fabs(fftarr->fftforw_magarr[2][idx][0]);	
		elecr += fabs(fftarr->fftforw_elecarr[idx][0]);	
		
		magzi += fabs(fftarr->fftforw_magarr[0][idx][1]);	
		magyi += fabs(fftarr->fftforw_magarr[1][idx][1]);	
		magxi += fabs(fftarr->fftforw_magarr[2][idx][1]);	
		eleci += fabs(fftarr->fftforw_elecarr[idx][1]);	
	}
	
	magxr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magyr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magzr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	elecr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);

	magxi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magyi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magzi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	eleci /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Average of space domain Green's function filters are (magxr, magxi), (magyr, magyi), (magzr, magzi), (elecr, eleci) = (%e,%e),(%e,%e),(%e,%e),(%e,%e)\n", magxr, magxi, magyr, magyi, magzr, magzi, elecr, eleci);
	
	fftw_execute(fftarr->fftforw_magplan[0]);
	fftw_execute(fftarr->fftforw_magplan[1]);
	fftw_execute(fftarr->fftforw_magplan[2]);
	fftw_execute(fftarr->fftforw_elecplan);

	magxr = 0; magyr = 0; magzr = 0; elecr = 0;
	magxi = 0; magyi = 0; magzi = 0; eleci = 0;
	for (i = 0; i < fftarr->z_num; i++)
	for (j = 0; j < fftarr->y_num; j++)
	for (k = 0; k < fftarr->x_num; k++)
	{
		idx = i*fftarr->y_num*fftarr->x_num + j*fftarr->x_num + k;	
	
		ObjPtr->MagFilt[i][j][k][2*0+0] = fftarr->fftforw_magarr[0][idx][0];	
		ObjPtr->MagFilt[i][j][k][2*1+0] = fftarr->fftforw_magarr[1][idx][0];	
		ObjPtr->MagFilt[i][j][k][2*2+0] = fftarr->fftforw_magarr[2][idx][0];	
		ObjPtr->ElecFilt[i][j][k][0] = fftarr->fftforw_elecarr[idx][0];	
		
		ObjPtr->MagFilt[i][j][k][2*0+1] = fftarr->fftforw_magarr[0][idx][1];	
		ObjPtr->MagFilt[i][j][k][2*1+1] = fftarr->fftforw_magarr[1][idx][1];	
		ObjPtr->MagFilt[i][j][k][2*2+1] = fftarr->fftforw_magarr[2][idx][1];	
		ObjPtr->ElecFilt[i][j][k][1] = fftarr->fftforw_elecarr[idx][1];

/*		printf("Freq i = %d, j = %d, k = %d, idx = %d, fft (mag,elec) = (%e,%e,%e,%e)\n", i, j, k, idx, fftarr->fftforw_magarr[0][idx][0], fftarr->fftforw_magarr[1][idx][0], fftarr->fftforw_magarr[2][idx][0], fftarr->fftforw_elecarr[idx][0]);*/

		magzr += fabs(fftarr->fftforw_magarr[0][idx][0]);	
		magyr += fabs(fftarr->fftforw_magarr[1][idx][0]);	
		magxr += fabs(fftarr->fftforw_magarr[2][idx][0]);	
		elecr += fabs(fftarr->fftforw_elecarr[idx][0]);	
		
		magzi += fabs(fftarr->fftforw_magarr[0][idx][1]);	
		magyi += fabs(fftarr->fftforw_magarr[1][idx][1]);	
		magxi += fabs(fftarr->fftforw_magarr[2][idx][1]);	
		eleci += fabs(fftarr->fftforw_elecarr[idx][1]);	
		
/*		magzr += fabs(ObjPtr->MagFilt[i][j][k][0][0]);	
		magyr += fabs(ObjPtr->MagFilt[i][j][k][1][0]);	
		magxr += fabs(ObjPtr->MagFilt[i][j][k][2][0]);	
		elecr += fabs(ObjPtr->ElecFilt[i][j][k][0]);	
		
		magzi += fabs(ObjPtr->MagFilt[i][j][k][0][1]);	
		magyi += fabs(ObjPtr->MagFilt[i][j][k][1][1]);	
		magxi += fabs(ObjPtr->MagFilt[i][j][k][2][1]);	
		eleci += fabs(ObjPtr->ElecFilt[i][j][k][1]);*/	
	}

	magxr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magyr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magzr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	elecr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);

	magxi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magyi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magzi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	eleci /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);

	
	Write2Bin ("mag_freq_resp", fftarr->z_num, fftarr->y_num, fftarr->x_num, 6, sizeof(Real_arr_t), &(ObjPtr->MagFilt[0][0][0][0]), InpPtr->debug_file_ptr);
	Write2Bin ("elec_freq_resp", fftarr->z_num, fftarr->y_num, fftarr->x_num, 2, sizeof(Real_arr_t), &(ObjPtr->ElecFilt[0][0][0][0]), InpPtr->debug_file_ptr);

	int dimTiff[4];
    	if (InpPtr->Write2Tiff == 1)
	{
    		dimTiff[0] = fftarr->z_num; dimTiff[1] = fftarr->y_num; dimTiff[2] = fftarr->x_num; dimTiff[3] = 6;
    		WriteMultiDimArray2Tiff ("mag_freq_resp", dimTiff, 0, 3, 1, 2, &(ObjPtr->MagFilt[0][0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr);
    		dimTiff[0] = fftarr->z_num; dimTiff[1] = fftarr->y_num; dimTiff[2] = fftarr->x_num; dimTiff[3] = 2;
    		WriteMultiDimArray2Tiff ("elec_freq_resp", dimTiff, 0, 3, 1, 2, &(ObjPtr->ElecFilt[0][0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr);
	}

	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Average of freq domain Green's function filters are (magxr, magxi), (magyr, magyi), (magzr, magzi), (elecr, eleci) = (%e,%e),(%e,%e),(%e,%e),(%e,%e)\n", magxr, magxi, magyr, magyi, magzr, magzi, elecr, eleci);
}


/*Initializes the variables in the three major structures used throughout the code -
Sinogram, ScannedObject, TomoInputs. It also allocates memory for several variables.*/
int32_t initStructures (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, int32_t mult_idx, int32_t mult_xy[], int32_t mult_z[], float *data_unflip_x, float *data_flip_x, float *data_unflip_y, float *data_flip_y, float *proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, Real_t vox_wid, Real_t rot_center, Real_t mag_sigma, Real_t mag_c, Real_t elec_sigma, Real_t elec_c, Real_t convg_thresh)
{
	int flag = 0, i;

	/*MPI node number and total node count parameters*/
	/*MPI_Comm_size(MPI_COMM_WORLD, &(InpPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(InpPtr->node_rank));*/
	InpPtr->node_num = 1; InpPtr->node_rank = 0;

	ObjPtr->Mag_Sigma[0] = mag_sigma;
	ObjPtr->Mag_Sigma[1] = mag_sigma;
	ObjPtr->Mag_Sigma[2] = mag_sigma;
	ObjPtr->Mag_C[0] = mag_c;
	ObjPtr->Mag_C[1] = mag_c;
	ObjPtr->Mag_C[2] = mag_c;
	
	ObjPtr->Elec_Sigma = elec_sigma;
	ObjPtr->Elec_C = elec_c;

	InpPtr->Weight = 1;
	
	ObjPtr->mult_xy = mult_xy[mult_idx];
	ObjPtr->mult_z = mult_z[mult_idx];		
	SinoPtr->Length_R = vox_wid*proj_cols;
	SinoPtr->Length_T = vox_wid*proj_rows;
	InpPtr->StopThreshold = convg_thresh;
	InpPtr->RotCenter = rot_center;
	InpPtr->alpha = OVER_RELAXATION_FACTOR;
	if (mult_idx == 0)
		InpPtr->initICD = 0;
	else if (mult_z[mult_idx] == mult_z[mult_idx-1]) 
		InpPtr->initICD = 2;
	else if (mult_z[mult_idx-1]/mult_z[mult_idx] == 2)
		InpPtr->initICD = 3;
	else
		sentinel(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Multi-resolution scaling is not supported");
		
	InpPtr->Write2Tiff = ENABLE_TIFF_WRITES;
	SinoPtr->N_p = proj_num;	
	SinoPtr->N_r = proj_cols;
	InpPtr->cost_thresh = COST_CONVG_THRESHOLD;	
	InpPtr->radius_obj = vox_wid*proj_cols;	
	SinoPtr->total_t_slices = proj_rows;
	InpPtr->no_NHICD = NO_NHICD;	
	InpPtr->WritePerIter = WRITE_EVERY_ITER;
		
	InpPtr->initMagUpMap = 0;
	if (mult_idx > 0)
	{
		InpPtr->initMagUpMap = 1;
	}

	/*Initializing Sinogram parameters*/
	int32_t j, k, idx;

	SinoPtr->Length_T = SinoPtr->Length_T/InpPtr->node_num;
	SinoPtr->N_t = SinoPtr->total_t_slices/InpPtr->node_num;
	
	SinoPtr->Data_Unflip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
	SinoPtr->Data_Flip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);

#ifdef VFET_TWO_AXES
	SinoPtr->Data_Unflip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);
	SinoPtr->Data_Flip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->N_p, SinoPtr->N_r, SinoPtr->N_t);

	for (i = 0; i < SinoPtr->N_p; i++)
	for (j = 0; j < SinoPtr->N_r; j++)
	for (k = 0; k < SinoPtr->N_t; k++)
	{
		idx = i*SinoPtr->N_t*SinoPtr->N_r + k*SinoPtr->N_r + j;
		SinoPtr->Data_Unflip_y[i][j][k] = data_unflip_y[idx];
		SinoPtr->Data_Flip_y[i][j][k] = data_flip_y[idx];
	}
#endif
	
	for (i = 0; i < SinoPtr->N_p; i++)
	for (j = 0; j < SinoPtr->N_r; j++)
	for (k = 0; k < SinoPtr->N_t; k++)
	{
		idx = i*SinoPtr->N_t*SinoPtr->N_r + j*SinoPtr->N_t + k;
		SinoPtr->Data_Unflip_x[i][j][k] = data_unflip_x[idx];
		SinoPtr->Data_Flip_x[i][j][k] = data_flip_x[idx];
	}

	int dimTiff[4];
    	dimTiff[0] = 1; dimTiff[1] = SinoPtr->N_p; dimTiff[2] = SinoPtr->N_r; dimTiff[3] = SinoPtr->N_t;
    	if (InpPtr->Write2Tiff == 1)
	{
    		if (WriteMultiDimArray2Tiff (DATA_UNFLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->Data_Unflip_x[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
    		if (WriteMultiDimArray2Tiff (DATA_FLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->Data_Flip_x[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
#ifdef VFET_TWO_AXES
    		if (WriteMultiDimArray2Tiff (DATA_UNFLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->Data_Unflip_y[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
    		if (WriteMultiDimArray2Tiff (DATA_FLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->Data_Flip_y[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
#endif
	}	
	
	SinoPtr->delta_r = SinoPtr->Length_R/(SinoPtr->N_r);
	SinoPtr->delta_t = SinoPtr->Length_T/(SinoPtr->N_t);
	SinoPtr->R0 = -InpPtr->RotCenter*SinoPtr->delta_r;
	SinoPtr->RMax = (SinoPtr->N_r-InpPtr->RotCenter)*SinoPtr->delta_r;
	SinoPtr->T0 = -SinoPtr->Length_T/2.0;
	SinoPtr->TMax = SinoPtr->Length_T/2.0;
	
	/*Initializing parameters of the object to be reconstructed*/
	ObjPtr->Length_X = SinoPtr->Length_R;
    	ObjPtr->Length_Y = SinoPtr->Length_R;
	ObjPtr->Length_Z = SinoPtr->Length_T;
    	ObjPtr->N_x = (int32_t)(SinoPtr->N_r/ObjPtr->mult_xy);
	ObjPtr->N_y = (int32_t)(SinoPtr->N_r/ObjPtr->mult_xy);
	ObjPtr->N_z = (int32_t)(SinoPtr->N_t/ObjPtr->mult_z);	
	ObjPtr->delta_xy = ObjPtr->mult_xy*SinoPtr->delta_r;
	ObjPtr->delta_z = ObjPtr->mult_z*SinoPtr->delta_t;
	SinoPtr->z_overlap_num = ObjPtr->mult_z;

	if (ObjPtr->delta_xy != ObjPtr->delta_z)
		check_warn (InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Voxel width in x-y plane is not equal to that along z-axis. The spatial invariance of prior does not hold.\n");

	ObjPtr->x0 = SinoPtr->R0;
    	ObjPtr->z0 = SinoPtr->T0;
    	ObjPtr->y0 = -ObjPtr->Length_Y/2.0;
    	ObjPtr->BeamWidth = SinoPtr->delta_r; /*Weighting of the projections at different points of the detector*/

	ObjPtr->MagPotentials = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
	ObjPtr->ElecPotentials = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);
	ObjPtr->Magnetization = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
	ObjPtr->ChargeDensity = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);
    	ObjPtr->MagPotDual = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
    	ObjPtr->ElecPotDual = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);

	/*OffsetR is stepsize of the distance between center of voxel of the object and the detector pixel, at which projections are computed*/
	SinoPtr->OffsetR = (ObjPtr->delta_xy/sqrt(2.0)+SinoPtr->delta_r/2.0)/DETECTOR_RESPONSE_BINS;
	SinoPtr->OffsetT = ((ObjPtr->delta_z/2) + SinoPtr->delta_t/2)/DETECTOR_RESPONSE_BINS;

	InpPtr->num_threads = omp_get_max_threads();
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Maximum number of openmp threads is %d\n", InpPtr->num_threads);
	if (InpPtr->num_threads <= 1)
		check_warn(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "The maximum number of threads is less than or equal to 1.\n");	
	InpPtr->num_z_blocks = InpPtr->num_threads;
	if (InpPtr->num_z_blocks < 2)
		InpPtr->num_z_blocks = 2;
	else if (InpPtr->num_z_blocks > ObjPtr->N_z)
		InpPtr->num_z_blocks = ObjPtr->N_z;
	InpPtr->num_z_blocks = (InpPtr->num_z_blocks/2)*2; /*Round down to the nearest even integer*/
	
	InpPtr->prevnum_z_blocks = InpPtr->num_threads;
	if (InpPtr->prevnum_z_blocks < 2)
		InpPtr->prevnum_z_blocks = 2;
	else 
	{
		if (InpPtr->initICD == 3 && InpPtr->prevnum_z_blocks > ObjPtr->N_z/2)
			InpPtr->prevnum_z_blocks = ObjPtr->N_z/2;
		else if (InpPtr->prevnum_z_blocks > ObjPtr->N_z)
			InpPtr->prevnum_z_blocks = ObjPtr->N_z;
	}
	InpPtr->prevnum_z_blocks = (InpPtr->prevnum_z_blocks/2)*2; /*Round down to the nearest even integer*/

/*	InpPtr->BoundaryFlag = (uint8_t***)multialloc(sizeof(uint8_t), 3, 3, 3, 3);*/
        InpPtr->x_rand_select = (int32_t**)multialloc(sizeof(int32_t), 2, InpPtr->num_z_blocks, ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->y_rand_select = (int32_t**)multialloc(sizeof(int32_t), 2, InpPtr->num_z_blocks, ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->x_NHICD_select = (int32_t**)multialloc(sizeof(int32_t), 2, InpPtr->num_z_blocks, ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->y_NHICD_select = (int32_t**)multialloc(sizeof(int32_t), 2, InpPtr->num_z_blocks, ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->UpdateSelectNum = (int32_t*)multialloc(sizeof(int32_t), 1, InpPtr->num_z_blocks);
        InpPtr->NHICDSelectNum = (int32_t*)multialloc(sizeof(int32_t), 1, InpPtr->num_z_blocks);

	ObjPtr->NHICD_Iterations = 10;

	check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Number of z blocks is %d\n", InpPtr->num_z_blocks);
	check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Number of z blocks in previous multi-resolution stage is %d\n", InpPtr->prevnum_z_blocks);

	SinoPtr->ViewPtr = (Real_arr_t*)get_spc(proj_num, sizeof(Real_arr_t));
	for (i = 0; i < proj_num; i++)
		SinoPtr->ViewPtr[i] = proj_angles[i];

	/*TomoInputs holds the input parameters and some miscellaneous variables*/
	for (i = 0; i < 3; i++)
	{
		InpPtr->Mag_Sigma_Q[i] = pow((ObjPtr->Mag_Sigma[i]*ObjPtr->mult_xy),MRF_Q);
		InpPtr->Mag_Sigma_Q_P[i] = pow(ObjPtr->Mag_Sigma[i]*ObjPtr->mult_xy,MRF_Q-MRF_P);	
	}

	InpPtr->Elec_Sigma_Q = pow((ObjPtr->Elec_Sigma*ObjPtr->mult_xy),MRF_Q);
	InpPtr->Elec_Sigma_Q_P = pow(ObjPtr->Elec_Sigma*ObjPtr->mult_xy,MRF_Q-MRF_P);	
	initFilter (ObjPtr, InpPtr);
	
	calculateSinCos (SinoPtr, InpPtr);
	
	InpPtr->ADMM_mu = 1;	

	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "The ADMM mu is %f.\n", InpPtr->ADMM_mu);
		
	InpPtr->NumIter = MAX_NUM_ITERATIONS;
	InpPtr->Head_MaxIter = 20;
	InpPtr->DensUpdate_MaxIter = 100;
	
	/*InpPtr->NumIter = 2;
	InpPtr->Head_MaxIter = 2;
	InpPtr->DensUpdate_MaxIter = 2;*/
	
	InpPtr->Head_threshold = 1;
	InpPtr->DensUpdate_thresh = convg_thresh;

/*#ifdef INIT_GROUND_TRUTH_PHANTOM
	ObjPtr->ElecPotGndTruth = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, PHANTOM_Z_SIZE, PHANTOM_Y_SIZE, PHANTOM_X_SIZE);
	ObjPtr->MagPotGndTruth = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, PHANTOM_Z_SIZE, PHANTOM_Y_SIZE, PHANTOM_X_SIZE, 3);
	size = PHANTOM_Z_SIZE*PHANTOM_Y_SIZE*PHANTOM_X_SIZE;
	if (read_SharedBinFile_At (PHANTOM_ELECOBJECT_FILENAME, &(ObjPtr->ElecPotGndTruth[0][0][0]), InpPtr->node_rank*size, size, InpPtr->debug_file_ptr)) flag = -1;
	if (read_SharedBinFile_At (PHANTOM_MAGOBJECT_FILENAME, &(ObjPtr->MagPotGndTruth[0][0][0][0]), InpPtr->node_rank*size*3, size*3, InpPtr->debug_file_ptr)) flag = -1;
#endif*/

	check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Initialized the structures, Sinogram and ScannedObject\n");
	
	check_error(SinoPtr->N_t % (int32_t)ObjPtr->mult_z != 0, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Cannot do reconstruction since mult_z = %d does not divide %d\n", (int32_t)ObjPtr->mult_z, SinoPtr->N_t);
	check_error(SinoPtr->N_r % (int32_t)ObjPtr->mult_xy != 0, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Cannot do reconstruction since mult_xy = %d does not divide %d\n", (int32_t)ObjPtr->mult_xy, SinoPtr->N_r);

	/*InpPtr->MagPhaseMultiple = 0.01;
	InpPtr->ElecPhaseMultiple = 0.01;*/
	
	InpPtr->MagPhaseMultiple = -3.794e-6*SinoPtr->delta_r*SinoPtr->delta_r;
	InpPtr->ElecPhaseMultiple = 0.0364*SinoPtr->delta_r;

	fftptr->z_num = 2*ObjPtr->N_z;
	fftptr->y_num = 2*ObjPtr->N_y;
	fftptr->x_num = 2*ObjPtr->N_x;
	fftptr->x0 = ObjPtr->N_x/2;
	fftptr->y0 = ObjPtr->N_y/2;
	fftptr->z0 = ObjPtr->N_z/2;
	
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Initializing FFT arrays and plans.\n");
	
	fftptr->fftforw_magarr = (fftw_complex**)get_spc(3, sizeof(fftw_complex*));
	fftptr->fftback_magarr = (fftw_complex**)get_spc(3, sizeof(fftw_complex*));
	fftptr->fftforw_magplan = (fftw_plan*)get_spc(3, sizeof(fftw_plan));
	fftptr->fftback_magplan = (fftw_plan*)get_spc(3, sizeof(fftw_plan));
	for (i = 0; i < 3; i++)	
	{
		fftptr->fftforw_magarr[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
		fftptr->fftforw_magplan[i] = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftforw_magarr[i], fftptr->fftforw_magarr[i], FFTW_FORWARD, FFTW_ESTIMATE);
		fftptr->fftback_magarr[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
		fftptr->fftback_magplan[i] = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftback_magarr[i], fftptr->fftback_magarr[i], FFTW_BACKWARD, FFTW_ESTIMATE);
	}
	
	fftptr->fftforw_elecarr = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
	fftptr->fftforw_elecplan = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftforw_elecarr, fftptr->fftforw_elecarr, FFTW_FORWARD, FFTW_ESTIMATE);
	fftptr->fftback_elecarr = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
	fftptr->fftback_elecplan = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftback_elecarr, fftptr->fftback_elecarr, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	ObjPtr->MagFilt = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, fftptr->z_num, fftptr->y_num, fftptr->x_num, 6);	
	ObjPtr->ElecFilt = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, fftptr->z_num, fftptr->y_num, fftptr->x_num, 2);	

	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Done initializing FFT arrays and plans.\n");
	
	/*compute cross product filters*/
	initCrossProdFilter (ObjPtr, InpPtr, fftptr);

	return (flag);
error:
	return (-1);	
}

/*Free memory of several arrays*/
void freeMemory(Sinogram* SinoPtr, ScannedObject *ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr)
{
	int32_t i;

	for (i = 0; i < 3; i++)
	{
		fftw_destroy_plan(fftptr->fftforw_magplan[i]);
		fftw_destroy_plan(fftptr->fftback_magplan[i]);
		fftw_free(fftptr->fftforw_magarr[i]); 
		fftw_free(fftptr->fftback_magarr[i]);
	}

	fftw_destroy_plan(fftptr->fftforw_elecplan);
	fftw_destroy_plan(fftptr->fftback_elecplan);
	fftw_free(fftptr->fftforw_elecarr); 
	fftw_free(fftptr->fftback_elecarr);

	free(fftptr->fftforw_magarr);
	free(fftptr->fftback_magarr);
	free(fftptr->fftforw_magplan);
	free(fftptr->fftback_magplan);

	multifree(ObjPtr->MagFilt, 4);
	multifree(ObjPtr->ElecFilt, 4);

/*#ifdef INIT_GROUND_TRUTH_PHANTOM
	multifree(ObjPtr->MagPotGndTruth,4);
	multifree(ObjPtr->ElecPotGndTruth,3);
#endif*/
	multifree(ObjPtr->MagPotentials,4);
	multifree(ObjPtr->ElecPotentials,3);
	multifree(ObjPtr->Magnetization,4);
	multifree(ObjPtr->ChargeDensity,3);
	multifree(ObjPtr->MagPotDual,4);
	multifree(ObjPtr->ElecPotDual,3);
	if (SinoPtr->ViewPtr) free(SinoPtr->ViewPtr);

	if (SinoPtr->Data_Unflip_x) multifree(SinoPtr->Data_Unflip_x,3);
	if (SinoPtr->Data_Flip_x) multifree(SinoPtr->Data_Flip_x,3);
#ifdef VFET_TWO_AXES
	if (SinoPtr->Data_Unflip_y) multifree(SinoPtr->Data_Unflip_y,3);
	if (SinoPtr->Data_Flip_y) multifree(SinoPtr->Data_Flip_y,3);
#endif
	
	if (InpPtr->x_rand_select) multifree(InpPtr->x_rand_select,2);
	if (InpPtr->y_rand_select) multifree(InpPtr->y_rand_select,2);
	if (InpPtr->x_NHICD_select) multifree(InpPtr->x_NHICD_select,2);
	if (InpPtr->y_NHICD_select) multifree(InpPtr->y_NHICD_select,2);
	if (InpPtr->UpdateSelectNum) multifree(InpPtr->UpdateSelectNum,1);
	if (InpPtr->NHICDSelectNum) multifree(InpPtr->NHICDSelectNum,1);
	if (SinoPtr->cosine) free(SinoPtr->cosine);
	if (SinoPtr->sine) free(SinoPtr->sine);
}


int32_t initPhantomStructures (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, float *proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, Real_t vox_wid, Real_t rot_center)
{
	int i;
	
	/*MPI_Comm_size(MPI_COMM_WORLD, &(InpPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(InpPtr->node_rank));*/

	InpPtr->node_num = 1;
	InpPtr->node_rank = 0;
	
	SinoPtr->Length_R = vox_wid*proj_cols;
	SinoPtr->Length_T = vox_wid*proj_rows;
	InpPtr->RotCenter = rot_center*(PHANTOM_XY_SIZE/proj_cols);
		
	InpPtr->Write2Tiff = ENABLE_TIFF_WRITES;
	SinoPtr->N_p = proj_num;	
	SinoPtr->N_r = proj_cols;
	SinoPtr->total_t_slices = proj_rows;

	SinoPtr->Length_T = SinoPtr->Length_T/InpPtr->node_num;
	SinoPtr->N_t = SinoPtr->total_t_slices/InpPtr->node_num;
	
	SinoPtr->delta_r = SinoPtr->Length_R/(SinoPtr->N_r);
	SinoPtr->delta_t = SinoPtr->Length_T/(SinoPtr->N_t);
	SinoPtr->R0 = -InpPtr->RotCenter*SinoPtr->delta_r;
	SinoPtr->RMax = (SinoPtr->N_r-InpPtr->RotCenter)*SinoPtr->delta_r;
	SinoPtr->T0 = -SinoPtr->Length_T/2.0;
	SinoPtr->TMax = SinoPtr->Length_T/2.0;
	
	/*Initializing parameters of the object to be reconstructed*/
	ObjPtr->Length_X = SinoPtr->Length_R;
    	ObjPtr->Length_Y = SinoPtr->Length_R;
	ObjPtr->Length_Z = SinoPtr->Length_T;
	
    	ObjPtr->N_x = (int32_t)(PHANTOM_XY_SIZE);
	ObjPtr->N_y = (int32_t)(PHANTOM_XY_SIZE);
	ObjPtr->N_z = (int32_t)(PHANTOM_Z_SIZE);	
	ObjPtr->delta_xy = SinoPtr->Length_R/PHANTOM_XY_SIZE;
	ObjPtr->delta_z = SinoPtr->Length_T/PHANTOM_Z_SIZE*InpPtr->node_num;

	if (ObjPtr->delta_xy != ObjPtr->delta_z)
		check_warn (InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Voxel width in x-y plane is not equal to that along z-axis. The spatial invariance of prior does not hold.\n");

	ObjPtr->x0 = SinoPtr->R0;
    	ObjPtr->z0 = SinoPtr->T0;
    	ObjPtr->y0 = -ObjPtr->Length_Y/2.0;
    	ObjPtr->BeamWidth = SinoPtr->delta_r; /*Weighting of the projections at different points of the detector*/
/*	ObjPtr->Object = (Real_t****)multialloc(sizeof(Real_t), 4, ObjPtr->N_time, ObjPtr->N_y, ObjPtr->N_x, ObjPtr->N_z);*/
	/*OffsetR is stepsize of the distance between center of voxel of the object and the detector pixel, at which projections are computed*/
	SinoPtr->OffsetR = (ObjPtr->delta_xy/sqrt(2.0) + SinoPtr->delta_r/2.0)/DETECTOR_RESPONSE_BINS;
	SinoPtr->OffsetT = ((ObjPtr->delta_z/2) + SinoPtr->delta_t/2)/DETECTOR_RESPONSE_BINS;

	InpPtr->num_threads = omp_get_max_threads();
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Maximum number of openmp threads is %d\n", InpPtr->num_threads);
	if (InpPtr->num_threads <= 1)
		check_warn(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "The maximum number of threads is less than or equal to 1.\n");	
	
	SinoPtr->ViewPtr = (Real_arr_t*)get_spc(proj_num, sizeof(Real_arr_t));
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Projection angles are - ");
	for (i = 0; i < proj_num; i++)
	{
		SinoPtr->ViewPtr[i] = proj_angles[i];
		check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "(%d,%f)", i, SinoPtr->ViewPtr[i]);
	}
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "\n");

	calculateSinCos (SinoPtr, InpPtr);
	
	/*InpPtr->MagPhaseMultiple = -0.001517;
	InpPtr->ElecPhaseMultiple = 0.007288;*/
	InpPtr->MagPhaseMultiple = 0.01; /*Gauss^-1 px^-2*/
	InpPtr->ElecPhaseMultiple = 0.01; /*V^-1 px^-1*/
	
	fftptr->z_num = 2*ObjPtr->N_z;
	fftptr->y_num = 2*ObjPtr->N_y;
	fftptr->x_num = 2*ObjPtr->N_x;
	fftptr->x0 = ObjPtr->N_x/2;
	fftptr->y0 = ObjPtr->N_y/2;
	fftptr->z0 = ObjPtr->N_z/2;
	
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Initializing FFT arrays and plans.\n");
	
	fftptr->fftforw_magarr = (fftw_complex**)get_spc(3, sizeof(fftw_complex*));
	fftptr->fftback_magarr = (fftw_complex**)get_spc(3, sizeof(fftw_complex*));
	fftptr->fftforw_magplan = (fftw_plan*)get_spc(3, sizeof(fftw_plan));
	fftptr->fftback_magplan = (fftw_plan*)get_spc(3, sizeof(fftw_plan));
	for (i = 0; i < 3; i++)	
	{
		fftptr->fftforw_magarr[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
		fftptr->fftforw_magplan[i] = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftforw_magarr[i], fftptr->fftforw_magarr[i], FFTW_FORWARD, FFTW_ESTIMATE);
		fftptr->fftback_magarr[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
		fftptr->fftback_magplan[i] = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftback_magarr[i], fftptr->fftback_magarr[i], FFTW_BACKWARD, FFTW_ESTIMATE);
	}
	
	fftptr->fftforw_elecarr = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
	fftptr->fftforw_elecplan = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftforw_elecarr, fftptr->fftforw_elecarr, FFTW_FORWARD, FFTW_ESTIMATE);
	fftptr->fftback_elecarr = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftptr->z_num*fftptr->y_num*fftptr->x_num);
	fftptr->fftback_elecplan = fftw_plan_dft_3d(fftptr->z_num, fftptr->y_num, fftptr->x_num, fftptr->fftback_elecarr, fftptr->fftback_elecarr, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	ObjPtr->MagFilt = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, fftptr->z_num, fftptr->y_num, fftptr->x_num, 6);	
	ObjPtr->ElecFilt = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, fftptr->z_num, fftptr->y_num, fftptr->x_num, 2);	

	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Done initializing FFT arrays and plans.\n");
	
	/*compute cross product filters*/
	initCrossProdFilter (ObjPtr, InpPtr, fftptr);

	check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Initialized the structures, Sinogram and ScannedObject\n");
	
	return (0);
}

/*Free memory of several arrays*/
void freePhantomMemory(Sinogram* SinoPtr, ScannedObject *ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr)
{
	int32_t i;

	for (i = 0; i < 3; i++)
	{
		fftw_destroy_plan(fftptr->fftforw_magplan[i]);
		fftw_destroy_plan(fftptr->fftback_magplan[i]);
		fftw_free(fftptr->fftforw_magarr[i]); 
		fftw_free(fftptr->fftback_magarr[i]);
	}

	fftw_destroy_plan(fftptr->fftforw_elecplan);
	fftw_destroy_plan(fftptr->fftback_elecplan);
	fftw_free(fftptr->fftforw_elecarr); 
	fftw_free(fftptr->fftback_elecarr);

	free(fftptr->fftforw_magarr);
	free(fftptr->fftback_magarr);
	free(fftptr->fftforw_magplan);
	free(fftptr->fftback_magplan);

	multifree(ObjPtr->MagFilt, 4);
	multifree(ObjPtr->ElecFilt, 4);

/*	int32_t i;
	for (i=0; i<ObjPtr->N_time; i++)
		{if (ObjPtr->ProjIdxPtr[i]) free(ObjPtr->ProjIdxPtr[i]);}
	
	if (ObjPtr->ProjIdxPtr) free(ObjPtr->ProjIdxPtr);
	if (ObjPtr->ProjNum) free(ObjPtr->ProjNum);
*/
/*	for (i = 0; i < ObjPtr->N_time; i++)
		{if (ObjPtr->Object[i]) multifree(ObjPtr->Object[i],3);}
	if (ObjPtr->Object) free(ObjPtr->Object);*/
/*	multifree(ObjPtr->Object, 4);*/
	
/*	if (SinoPtr->Measurements) multifree(SinoPtr->Measurements,3);
	if (SinoPtr->MagTomoAux) multifree(SinoPtr->MagTomoAux,3);
	if (SinoPtr->MagTomoDual) multifree(SinoPtr->MagTomoDual,3);
	if (SinoPtr->PhaseTomoAux) multifree(SinoPtr->PhaseTomoAux,3);
	if (SinoPtr->PhaseTomoDual) multifree(SinoPtr->PhaseTomoDual,3);
*/
/*	if (InpPtr->x_rand_select) multifree(InpPtr->x_rand_select,3);
	if (InpPtr->y_rand_select) multifree(InpPtr->y_rand_select,3);
	if (InpPtr->x_NHICD_select) multifree(InpPtr->x_NHICD_select,3);
	if (InpPtr->y_NHICD_select) multifree(InpPtr->y_NHICD_select,3);
	if (InpPtr->UpdateSelectNum) multifree(InpPtr->UpdateSelectNum,2);
	if (InpPtr->NHICDSelectNum) multifree(InpPtr->NHICDSelectNum,2);
	if (InpPtr->Weight) multifree(InpPtr->Weight,3);	*/
	if (SinoPtr->ViewPtr) free(SinoPtr->ViewPtr);
	if (SinoPtr->cosine) free(SinoPtr->cosine);
	if (SinoPtr->sine) free(SinoPtr->sine);
}


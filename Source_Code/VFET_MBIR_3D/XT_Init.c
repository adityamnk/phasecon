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
	prior_const = ObjPtr->mult_xyz*ObjPtr->mult_xyz*ObjPtr->mult_xyz;
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

  SinoPtr->cosine_x=(Real_t*)get_spc(SinoPtr->Nx_p, sizeof(Real_t));
  SinoPtr->sine_x=(Real_t*)get_spc(SinoPtr->Nx_p, sizeof(Real_t));
  SinoPtr->cosine_y=(Real_t*)get_spc(SinoPtr->Ny_p, sizeof(Real_t));
  SinoPtr->sine_y=(Real_t*)get_spc(SinoPtr->Ny_p, sizeof(Real_t));

  for(i=0;i<SinoPtr->Nx_p;i++)
  {
    SinoPtr->cosine_x[i]=cos(-SinoPtr->ViewPtr_x[i]);
    SinoPtr->sine_x[i]=sin(-SinoPtr->ViewPtr_x[i]);
  }

  for(i=0;i<SinoPtr->Ny_p;i++)
  {
    SinoPtr->cosine_y[i]=cos(SinoPtr->ViewPtr_y[i]);
    SinoPtr->sine_y[i]=sin(SinoPtr->ViewPtr_y[i]);
  }
  check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "calculateSinCos: Calculated sines and cosines of angles of rotation\n");
}


void initCrossProdFilter (ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftarr)
{
	int32_t i, j, k, idx_i, idx_j, idx_k, idx;
	Real_t win, delta, magxr, magyr, magzr, magxi, magyi, magzi;

	magxr = 0; magyr = 0; magzr = 0;
	magxi = 0; magyi = 0; magzi = 0; 
	
	for (i = fftarr->z_num/2 - 1; i >= -fftarr->z_num/2; i--)
	for (j = fftarr->y_num/2 - 1; j >= -fftarr->y_num/2; j--)
	for (k = fftarr->x_num/2 - 1; k >= -fftarr->x_num/2; k--)
	{
		idx_i = (i + fftarr->z_num) % (fftarr->z_num);
		idx_j = (j + fftarr->y_num) % (fftarr->y_num);
		idx_k = (k + fftarr->x_num) % (fftarr->x_num);

		idx = idx_i*fftarr->y_num*fftarr->x_num	+ idx_j*fftarr->x_num + idx_k;	
	
		if (i != 0 || j != 0 || k != 0)
		{
			win = (0.54 + 0.46*cos(2*M_PI*i/(fftarr->z_num - 1)));
			win *= (0.54 + 0.46*cos(2*M_PI*j/(fftarr->y_num - 1)));
			win *= (0.54 + 0.46*cos(2*M_PI*k/(fftarr->x_num - 1)));
		
			win /= pow(sqrt((Real_t)(i*i + j*j + k*k)), 3);

			fftarr->fftforw_magarr[0][idx][0] = ObjPtr->delta_x*((Real_t)i)*win;
			fftarr->fftforw_magarr[1][idx][0] = ObjPtr->delta_y*((Real_t)j)*win;
			fftarr->fftforw_magarr[2][idx][0] = ObjPtr->delta_z*((Real_t)k)*win;
		
		}
		else
		{
			fftarr->fftforw_magarr[0][idx][0] = 0;
			fftarr->fftforw_magarr[1][idx][0] = 0;
			fftarr->fftforw_magarr[2][idx][0] = 0;
		}
		
		fftarr->fftforw_magarr[0][idx][1] = 0;
		fftarr->fftforw_magarr[1][idx][1] = 0;
		fftarr->fftforw_magarr[2][idx][1] = 0;
		
/*		printf("Space i = %d, j = %d, k = %d, idx_i = %d, idx_j = %d, idx_k = %d, idx = %d, fft (mag,elec) = ((%e,%e),(%e,%e),(%e,%e),(%e,%e))\n", i, j, k, idx_i, idx_j, idx_k, idx, fftarr->fftforw_magarr[0][idx][0], fftarr->fftforw_magarr[0][idx][1], fftarr->fftforw_magarr[1][idx][0], fftarr->fftforw_magarr[1][idx][1], fftarr->fftforw_magarr[2][idx][0], fftarr->fftforw_magarr[2][idx][1], fftarr->fftforw_elecarr[idx][0], fftarr->fftforw_elecarr[idx][1]);*/
		
		magzr += fabs(fftarr->fftforw_magarr[0][idx][0]);	
		magyr += fabs(fftarr->fftforw_magarr[1][idx][0]);	
		magxr += fabs(fftarr->fftforw_magarr[2][idx][0]);	
		
		magzi += fabs(fftarr->fftforw_magarr[0][idx][1]);	
		magyi += fabs(fftarr->fftforw_magarr[1][idx][1]);	
		magxi += fabs(fftarr->fftforw_magarr[2][idx][1]);	
		
	}
	
	magxr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magyr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magzr /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);

	magxi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magyi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magzi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Average of space domain Green's function filters are (magxr, magxi), (magyr, magyi), (magzr, magzi) = (%e,%e),(%e,%e),(%e,%e)\n", magxr, magxi, magyr, magyi, magzr, magzi);
	
	fftw_execute(fftarr->fftforw_magplan[0]);
	fftw_execute(fftarr->fftforw_magplan[1]);
	fftw_execute(fftarr->fftforw_magplan[2]);

	magxr = 0; magyr = 0; magzr = 0;
	magxi = 0; magyi = 0; magzi = 0;
	
	for (i = 0; i < fftarr->z_num; i++)
	for (j = 0; j < fftarr->y_num; j++)
	for (k = 0; k < fftarr->x_num; k++)
	{
		idx = i*fftarr->y_num*fftarr->x_num + j*fftarr->x_num + k;	
	
		ObjPtr->MagFilt[i][j][k][2*0+0] = fftarr->fftforw_magarr[0][idx][0];	
		ObjPtr->MagFilt[i][j][k][2*1+0] = fftarr->fftforw_magarr[1][idx][0];	
		ObjPtr->MagFilt[i][j][k][2*2+0] = fftarr->fftforw_magarr[2][idx][0];	
		
		ObjPtr->MagFilt[i][j][k][2*0+1] = fftarr->fftforw_magarr[0][idx][1];	
		ObjPtr->MagFilt[i][j][k][2*1+1] = fftarr->fftforw_magarr[1][idx][1];	
		ObjPtr->MagFilt[i][j][k][2*2+1] = fftarr->fftforw_magarr[2][idx][1];	

/*		printf("Freq i = %d, j = %d, k = %d, idx = %d, fft (mag,elec) = (%e,%e,%e,%e)\n", i, j, k, idx, fftarr->fftforw_magarr[0][idx][0], fftarr->fftforw_magarr[1][idx][0], fftarr->fftforw_magarr[2][idx][0], fftarr->fftforw_elecarr[idx][0]);*/

		magzr += fabs(fftarr->fftforw_magarr[0][idx][0]);	
		magyr += fabs(fftarr->fftforw_magarr[1][idx][0]);	
		magxr += fabs(fftarr->fftforw_magarr[2][idx][0]);	
		
		magzi += fabs(fftarr->fftforw_magarr[0][idx][1]);	
		magyi += fabs(fftarr->fftforw_magarr[1][idx][1]);	
		magxi += fabs(fftarr->fftforw_magarr[2][idx][1]);	
		
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

	magxi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magyi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	magzi /= (fftarr->z_num*fftarr->y_num*fftarr->x_num);
	
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Average of freq domain Green's function filters are (magxr, magxi), (magyr, magyi), (magzr, magzi) = (%e,%e),(%e,%e),(%e,%e)\n", magxr, magxi, magyr, magyi, magzr, magzi);

	Write2Bin ("mag_freq_resp", fftarr->z_num, fftarr->y_num, fftarr->x_num, 6, sizeof(Real_arr_t), &(ObjPtr->MagFilt[0][0][0][0]), InpPtr->debug_file_ptr);

	/*int dimTiff[4];
    	if (InpPtr->Write2Tiff == 1)
	{
    		dimTiff[0] = fftarr->z_num; dimTiff[1] = fftarr->y_num; dimTiff[2] = fftarr->x_num; dimTiff[3] = 6;
    		WriteMultiDimArray2Tiff ("mag_freq_resp", dimTiff, 0, 3, 1, 2, &(ObjPtr->MagFilt[0][0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr);
    		dimTiff[0] = fftarr->z_num; dimTiff[1] = fftarr->y_num; dimTiff[2] = fftarr->x_num; dimTiff[3] = 2;
    		WriteMultiDimArray2Tiff ("elec_freq_resp", dimTiff, 0, 3, 1, 2, &(ObjPtr->ElecFilt[0][0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr);
	}*/

}


/*Initializes the variables in the three major structures used throughout the code -
Sinogram, ScannedObject, TomoInputs. It also allocates memory for several variables.*/
int32_t initStructures (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, int32_t mult_idx, int32_t mult_xyz[], float *data_unflip_x, float *data_unflip_y, float *proj_angles_x, float *proj_angles_y, int32_t proj_rows, int32_t proj_cols, int32_t proj_x_num, int32_t proj_y_num, int32_t x_widnum, int32_t y_widnum, int32_t z_widnum, Real_t vox_wid, Real_t qggmrf_sigma, Real_t qggmrf_c, Real_t convg_thresh, Real_t admm_mu, int32_t admm_maxiters, float data_var)
{
	int flag = 0, i;

	/*MPI node number and total node count parameters*/
	/*MPI_Comm_size(MPI_COMM_WORLD, &(InpPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(InpPtr->node_rank));*/
	InpPtr->node_num = 1; InpPtr->node_rank = 0;

	ObjPtr->Mag_Sigma[0] = qggmrf_sigma;
	ObjPtr->Mag_Sigma[1] = qggmrf_sigma;
	ObjPtr->Mag_Sigma[2] = qggmrf_sigma;
	ObjPtr->Mag_C[0] = qggmrf_c;
	ObjPtr->Mag_C[1] = qggmrf_c;
	ObjPtr->Mag_C[2] = qggmrf_c;

	InpPtr->Weight = 1.0/data_var;
	
	ObjPtr->mult_xyz = mult_xyz[mult_idx];
	SinoPtr->Length_R = vox_wid*proj_cols;
	SinoPtr->Length_T = vox_wid*proj_rows;
	InpPtr->StopThreshold = convg_thresh;
	InpPtr->alpha = OVER_RELAXATION_FACTOR;
	if (mult_idx == 0)
		InpPtr->initICD = 0;
	else if (mult_xyz[mult_idx-1]/mult_xyz[mult_idx] == 2)
		InpPtr->initICD = 3;
	else
		sentinel(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Multi-resolution scaling is not supported");
		
	InpPtr->Write2Tiff = ENABLE_TIFF_WRITES;
	SinoPtr->Nx_p = proj_x_num;	
	SinoPtr->Ny_p = proj_y_num;	
	SinoPtr->N_r = proj_cols;
	InpPtr->cost_thresh = COST_CONVG_THRESHOLD;	
/*	InpPtr->radius_obj = vox_wid*proj_cols;*/	
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
	
	SinoPtr->Data_Unflip_x = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->Nx_p, SinoPtr->N_r, SinoPtr->N_t);

	for (i = 0; i < SinoPtr->Nx_p; i++)
	for (j = 0; j < SinoPtr->N_r; j++)
	for (k = 0; k < SinoPtr->N_t; k++)
	{
		idx = i*SinoPtr->N_t*SinoPtr->N_r + j*SinoPtr->N_t + k;
		SinoPtr->Data_Unflip_x[i][j][k] = data_unflip_x[idx];
	}

#ifdef VFET_TWO_AXES
	SinoPtr->Data_Unflip_y = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinoPtr->Ny_p, SinoPtr->N_r, SinoPtr->N_t);

	for (i = 0; i < SinoPtr->Ny_p; i++)
	for (j = 0; j < SinoPtr->N_r; j++)
	for (k = 0; k < SinoPtr->N_t; k++)
	{
		idx = i*SinoPtr->N_t*SinoPtr->N_r + k*SinoPtr->N_r + j;
		SinoPtr->Data_Unflip_y[i][j][k] = data_unflip_y[idx];
	}
#endif
	
	int dimTiff[4];
    	if (InpPtr->Write2Tiff == 1)
	{
    		dimTiff[0] = 1; dimTiff[1] = SinoPtr->Nx_p; dimTiff[2] = SinoPtr->N_r; dimTiff[3] = SinoPtr->N_t;
    		if (WriteMultiDimArray2Tiff (DATA_UNFLIP_X_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->Data_Unflip_x[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
#ifdef VFET_TWO_AXES
    		dimTiff[0] = 1; dimTiff[1] = SinoPtr->Ny_p; dimTiff[2] = SinoPtr->N_r; dimTiff[3] = SinoPtr->N_t;
    		if (WriteMultiDimArray2Tiff (DATA_UNFLIP_Y_FILENAME, dimTiff, 0, 1, 2, 3, &(SinoPtr->Data_Unflip_y[0][0][0]), 0, 0, 1, InpPtr->debug_file_ptr)) goto error;
#endif
	}	
	
	SinoPtr->delta_r = SinoPtr->Length_R/(SinoPtr->N_r);
	SinoPtr->delta_t = SinoPtr->Length_T/(SinoPtr->N_t);
	SinoPtr->R0 = -SinoPtr->Length_R/2.0;
	SinoPtr->RMax = SinoPtr->Length_R/2.0;
	SinoPtr->T0 = -SinoPtr->Length_T/2.0;
	SinoPtr->TMax = SinoPtr->Length_T/2.0;
	
	/*Initializing parameters of the object to be reconstructed*/
	ObjPtr->Length_X = x_widnum*vox_wid;
    	ObjPtr->Length_Y = y_widnum*vox_wid;
	ObjPtr->Length_Z = z_widnum*vox_wid;
    	ObjPtr->N_x = (int32_t)(x_widnum/ObjPtr->mult_xyz);
	ObjPtr->N_y = (int32_t)(y_widnum/ObjPtr->mult_xyz);
	ObjPtr->N_z = (int32_t)(z_widnum/ObjPtr->mult_xyz);	
	ObjPtr->delta_x = ObjPtr->Length_X/ObjPtr->N_x;
	ObjPtr->delta_y = ObjPtr->Length_Y/ObjPtr->N_y;/*Subsequent code assumes delta_x = delta_y*/
	ObjPtr->delta_z = ObjPtr->Length_Z/ObjPtr->N_z;
	SinoPtr->z_overlap_num = ObjPtr->mult_xyz;

	ObjPtr->x0 = -ObjPtr->Length_X/2.0;
	ObjPtr->y0 = -ObjPtr->Length_Y/2.0;
	ObjPtr->z0 = -ObjPtr->Length_Z/2.0;
    	ObjPtr->BeamWidth = SinoPtr->delta_r; /*Weighting of the projections at different points of the detector*/

	ObjPtr->MagPotentials = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
	ObjPtr->Magnetization = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
    	ObjPtr->MagPotDual = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
    	ObjPtr->MagPotUpdateMap = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);

	/*OffsetR is stepsize of the distance between center of voxel of the object and the detector pixel, at which projections are computed*/
	SinoPtr->OffsetR = (ObjPtr->delta_x/sqrt(2.0)+SinoPtr->delta_r/2.0)/DETECTOR_RESPONSE_BINS;
	SinoPtr->OffsetT = ((ObjPtr->delta_z/2) + SinoPtr->delta_t/2)/DETECTOR_RESPONSE_BINS;

	InpPtr->num_threads = omp_get_max_threads();
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Maximum number of openmp threads is %d\n", InpPtr->num_threads);
	if (InpPtr->num_threads <= 1)
		check_warn(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "The maximum number of threads is less than or equal to 1.\n");	
	
/*	InpPtr->BoundaryFlag = (uint8_t***)multialloc(sizeof(uint8_t), 3, 3, 3, 3);*/
        InpPtr->x_rand_select = (int32_t*)multialloc(sizeof(int32_t), 1, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->y_rand_select = (int32_t*)multialloc(sizeof(int32_t), 1, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->z_rand_select = (int32_t*)multialloc(sizeof(int32_t), 1, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);

        InpPtr->x_NHICD_select = (int32_t*)multialloc(sizeof(int32_t), 1, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->y_NHICD_select = (int32_t*)multialloc(sizeof(int32_t), 1, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
        InpPtr->z_NHICD_select = (int32_t*)multialloc(sizeof(int32_t), 1, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);

	ObjPtr->NHICD_Iterations = 10;

	SinoPtr->ViewPtr_x = (Real_arr_t*)get_spc(proj_x_num, sizeof(Real_arr_t));
	SinoPtr->ViewPtr_y = (Real_arr_t*)get_spc(proj_y_num, sizeof(Real_arr_t));
	for (i = 0; i < proj_x_num; i++)
		SinoPtr->ViewPtr_x[i] = proj_angles_x[i];
	for (i = 0; i < proj_y_num; i++)
		SinoPtr->ViewPtr_y[i] = proj_angles_y[i];

	/*TomoInputs holds the input parameters and some miscellaneous variables*/
	for (i = 0; i < 3; i++)
	{
		InpPtr->Mag_Sigma_Q[i] = pow((ObjPtr->Mag_Sigma[i]*ObjPtr->mult_xyz),MRF_Q);
		InpPtr->Mag_Sigma_Q_P[i] = pow(ObjPtr->Mag_Sigma[i]*ObjPtr->mult_xyz,MRF_Q-MRF_P);	
	}

	initFilter (ObjPtr, InpPtr);
	
	calculateSinCos (SinoPtr, InpPtr);
	
	InpPtr->ADMM_mu = admm_mu;	

	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "The ADMM mu is %f.\n", InpPtr->ADMM_mu);
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Sigma_Q is (%f,%f,%f) and Sigma_Q_P is (%f,%f,%f).\n", InpPtr->Mag_Sigma_Q[0], InpPtr->Mag_Sigma_Q[1], InpPtr->Mag_Sigma_Q[2], InpPtr->Mag_Sigma_Q_P[0], InpPtr->Mag_Sigma_Q_P[1], InpPtr->Mag_Sigma_Q_P[2]);
		
	InpPtr->NumIter = MAX_NUM_ITERATIONS;
	InpPtr->Head_MaxIter = admm_maxiters;
	InpPtr->DensUpdate_MaxIter = 100;
	
	/*InpPtr->NumIter = 2;
	InpPtr->Head_MaxIter = 2;
	InpPtr->DensUpdate_MaxIter = 2;*/
	
	InpPtr->Head_threshold = 1;
	InpPtr->DensUpdate_thresh = convg_thresh;

#ifdef INIT_GROUND_TRUTH_PHANTOM
/*	ObjPtr->ElecPotGndTruth = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, PHANTOM_Z_SIZE, PHANTOM_Y_SIZE, PHANTOM_X_SIZE);*/
	ObjPtr->MagPotGndTruth = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, PHANTOM_Z_SIZE, PHANTOM_XY_SIZE, PHANTOM_XY_SIZE, 3);
	int32_t size = PHANTOM_Z_SIZE*PHANTOM_XY_SIZE*PHANTOM_XY_SIZE;
/*	if (read_SharedBinFile_At (PHANTOM_ELECOBJECT_FILENAME, &(ObjPtr->ElecPotGndTruth[0][0][0]), InpPtr->node_rank*size, size, InpPtr->debug_file_ptr)) flag = -1;*/
	Read4mBin (PHANTOM_MAGDENSITY_FILENAME, PHANTOM_Z_SIZE, PHANTOM_XY_SIZE, PHANTOM_XY_SIZE, 3, sizeof(Real_arr_t), &(ObjPtr->MagPotGndTruth[0][0][0][0]), InpPtr->debug_file_ptr);
#endif

	check_debug(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Initialized the structures, Sinogram and ScannedObject\n");
	
	check_error(SinoPtr->N_t % (int32_t)ObjPtr->mult_xyz != 0, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Cannot do reconstruction since mult_xyz = %d does not divide N_t = %d\n", (int32_t)ObjPtr->mult_xyz, SinoPtr->N_t);
	check_error(SinoPtr->N_r % (int32_t)ObjPtr->mult_xyz != 0, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Cannot do reconstruction since mult_xyz = %d does not divide N_r = %d\n", (int32_t)ObjPtr->mult_xyz, SinoPtr->N_r);

	InpPtr->MagPhaseMultiple = 1;
	InpPtr->ElecPhaseMultiple = 1;

	/*InpPtr->MagPhaseMultiple = -3.794e-6*SinoPtr->delta_r*SinoPtr->delta_r;
	InpPtr->MagPhaseMultiple = 0.01*SinoPtr->delta_r;
	InpPtr->ElecPhaseMultiple = 0.01*SinoPtr->delta_r;*/
	/*InpPtr->MagPhaseMultiple = 0.000003794;
	InpPtr->ElecPhaseMultiple = 0.0364;*/
	
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Mag and Elec pre multipliers to the phase are %e and %e\n.", InpPtr->MagPhaseMultiple, InpPtr->ElecPhaseMultiple);

	fftptr->z_num = ObjPtr->N_z;
	fftptr->y_num = ObjPtr->N_y;
	fftptr->x_num = ObjPtr->N_x;
	fftptr->x0 = 0;
	fftptr->y0 = 0;
	fftptr->z0 = 0;
	
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
	
	ObjPtr->MagFilt = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, fftptr->z_num, fftptr->y_num, fftptr->x_num, 6);	
	
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

	free(fftptr->fftforw_magarr);
	free(fftptr->fftback_magarr);
	free(fftptr->fftforw_magplan);
	free(fftptr->fftback_magplan);

	multifree(ObjPtr->MagFilt, 4);
	
	multifree(ObjPtr->MagPotentials,4);
	multifree(ObjPtr->Magnetization,4);
	multifree(ObjPtr->MagPotDual,4);
	multifree(ObjPtr->MagPotUpdateMap, 3);

#ifdef INIT_GROUND_TRUTH_PHANTOM
	multifree(ObjPtr->MagPotGndTruth,4);
/*	multifree(ObjPtr->ElecPotGndTruth,3);*/
#endif
	if (SinoPtr->ViewPtr_x) free(SinoPtr->ViewPtr_x);
	if (SinoPtr->ViewPtr_y) free(SinoPtr->ViewPtr_y);

	if (SinoPtr->Data_Unflip_x) multifree(SinoPtr->Data_Unflip_x,3);
#ifdef VFET_ELEC_RECON
	if (SinoPtr->Data_Flip_x) multifree(SinoPtr->Data_Flip_x,3);
#endif
#ifdef VFET_TWO_AXES
	if (SinoPtr->Data_Unflip_y) multifree(SinoPtr->Data_Unflip_y,3);
	#ifdef VFET_ELEC_RECON
	if (SinoPtr->Data_Flip_y) multifree(SinoPtr->Data_Flip_y,3);
	#endif
#endif
	
	if (InpPtr->x_rand_select) multifree(InpPtr->x_rand_select,1);
	if (InpPtr->y_rand_select) multifree(InpPtr->y_rand_select,1);
	if (InpPtr->z_rand_select) multifree(InpPtr->z_rand_select,1);
	if (InpPtr->x_NHICD_select) multifree(InpPtr->x_NHICD_select,1);
	if (InpPtr->y_NHICD_select) multifree(InpPtr->y_NHICD_select,1);
	if (InpPtr->z_NHICD_select) multifree(InpPtr->z_NHICD_select,1);
	if (SinoPtr->cosine_x) free(SinoPtr->cosine_x);
	if (SinoPtr->sine_x) free(SinoPtr->sine_x);
	if (SinoPtr->cosine_y) free(SinoPtr->cosine_y);
	if (SinoPtr->sine_y) free(SinoPtr->sine_y);
}


int32_t initPhantomStructures (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, float *proj_angles_x, float *proj_angles_y, int32_t proj_rows, int32_t proj_cols, int32_t proj_x_num, int32_t proj_y_num, float vox_wid, float data_var)
{
	int i;
	
	/*MPI_Comm_size(MPI_COMM_WORLD, &(InpPtr->node_num));
	MPI_Comm_rank(MPI_COMM_WORLD, &(InpPtr->node_rank));*/

	InpPtr->node_num = 1;
	InpPtr->node_rank = 0;

	InpPtr->Weight = 1.0/data_var;	
	SinoPtr->Length_R = vox_wid*proj_cols;
	SinoPtr->Length_T = vox_wid*proj_rows;
		
	InpPtr->Write2Tiff = ENABLE_TIFF_WRITES;
	SinoPtr->Nx_p = proj_x_num;	
	SinoPtr->Ny_p = proj_y_num;	
	SinoPtr->N_r = proj_cols;
	SinoPtr->total_t_slices = proj_rows;

	SinoPtr->Length_T = SinoPtr->Length_T/InpPtr->node_num;
	SinoPtr->N_t = SinoPtr->total_t_slices/InpPtr->node_num;
	
	SinoPtr->delta_r = SinoPtr->Length_R/(SinoPtr->N_r);
	SinoPtr->delta_t = SinoPtr->Length_T/(SinoPtr->N_t);
	SinoPtr->R0 = -SinoPtr->Length_R/2.0;
	SinoPtr->RMax = SinoPtr->Length_R/2.0;
	SinoPtr->T0 = -SinoPtr->Length_T/2.0;
	SinoPtr->TMax = SinoPtr->Length_T/2.0;
	
	/*Initializing parameters of the object to be reconstructed*/
	ObjPtr->delta_x = vox_wid/(PHANTOM_XY_SIZE/proj_cols);
	ObjPtr->delta_y = vox_wid/(PHANTOM_XY_SIZE/proj_cols);
	ObjPtr->delta_z = vox_wid/(PHANTOM_XY_SIZE/proj_cols);

	ObjPtr->Length_X = PHANTOM_XY_SIZE*ObjPtr->delta_x;
    	ObjPtr->Length_Y = PHANTOM_XY_SIZE*ObjPtr->delta_y;
	ObjPtr->Length_Z = PHANTOM_Z_SIZE*ObjPtr->delta_z;
	
    	ObjPtr->N_x = (int32_t)(PHANTOM_XY_SIZE);
	ObjPtr->N_y = (int32_t)(PHANTOM_XY_SIZE);
	ObjPtr->N_z = (int32_t)(PHANTOM_Z_SIZE);	

	ObjPtr->x0 = -ObjPtr->Length_X/2.0;
	ObjPtr->y0 = -ObjPtr->Length_Y/2.0;
	ObjPtr->z0 = -ObjPtr->Length_Z/2.0;
    	ObjPtr->BeamWidth = SinoPtr->delta_r; /*Weighting of the projections at different points of the detector*/
/*	ObjPtr->Object = (Real_t****)multialloc(sizeof(Real_t), 4, ObjPtr->N_time, ObjPtr->N_y, ObjPtr->N_x, ObjPtr->N_z);*/
	/*OffsetR is stepsize of the distance between center of voxel of the object and the detector pixel, at which projections are computed*/
	SinoPtr->OffsetR = (ObjPtr->delta_x/sqrt(2.0) + SinoPtr->delta_r/2.0)/DETECTOR_RESPONSE_BINS;
	SinoPtr->OffsetT = ((ObjPtr->delta_z/2) + SinoPtr->delta_t/2)/DETECTOR_RESPONSE_BINS;

	InpPtr->num_threads = omp_get_max_threads();
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Maximum number of openmp threads is %d\n", InpPtr->num_threads);
	if (InpPtr->num_threads <= 1)
		check_warn(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "The maximum number of threads is less than or equal to 1.\n");	
	
	SinoPtr->ViewPtr_x = (Real_arr_t*)get_spc(proj_x_num, sizeof(Real_arr_t));
	SinoPtr->ViewPtr_y = (Real_arr_t*)get_spc(proj_y_num, sizeof(Real_arr_t));
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Projection angles are - ");
	for (i = 0; i < proj_x_num; i++)
	{
		SinoPtr->ViewPtr_x[i] = proj_angles_x[i];
		check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "(%d,%f)", i, SinoPtr->ViewPtr_x[i]);
	}
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "\n");

	for (i = 0; i < proj_y_num; i++)
	{
		SinoPtr->ViewPtr_y[i] = proj_angles_y[i];
		check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "(%d,%f)", i, SinoPtr->ViewPtr_y[i]);
	}
	check_info(InpPtr->node_rank==0, InpPtr->debug_file_ptr, "\n");

	calculateSinCos (SinoPtr, InpPtr);
	
	/*InpPtr->MagPhaseMultiple = -0.001517;
	InpPtr->ElecPhaseMultiple = 0.007288;*/
	/*InpPtr->MagPhaseMultiple = -3.794e-6*SinoPtr->delta_r*SinoPtr->delta_r;*/
	/*InpPtr->MagPhaseMultiple = 0.01*SinoPtr->delta_r;
	InpPtr->ElecPhaseMultiple = 0.01*SinoPtr->delta_r;*/
	InpPtr->MagPhaseMultiple = 1;
	InpPtr->ElecPhaseMultiple = 1;
	
	fftptr->z_num = ObjPtr->N_z;
	fftptr->y_num = ObjPtr->N_y;
	fftptr->x_num = ObjPtr->N_x;
	fftptr->x0 = 0;
	fftptr->y0 = 0;
	fftptr->z0 = 0;
	
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
	
	ObjPtr->MagFilt = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, fftptr->z_num, fftptr->y_num, fftptr->x_num, 6);	

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

	free(fftptr->fftforw_magarr);
	free(fftptr->fftback_magarr);
	free(fftptr->fftforw_magplan);
	free(fftptr->fftback_magplan);

	multifree(ObjPtr->MagFilt, 4);

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
	if (SinoPtr->ViewPtr_x) free(SinoPtr->ViewPtr_x);
	if (SinoPtr->ViewPtr_y) free(SinoPtr->ViewPtr_y);
	if (SinoPtr->cosine_x) free(SinoPtr->cosine_x);
	if (SinoPtr->sine_x) free(SinoPtr->sine_x);
	if (SinoPtr->cosine_y) free(SinoPtr->cosine_y);
	if (SinoPtr->sine_y) free(SinoPtr->sine_y);
}


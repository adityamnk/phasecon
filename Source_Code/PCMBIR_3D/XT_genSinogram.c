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





/*#include <iostream>*/
/*#include "TiffUtilities.h"*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "XT_Structures.h"
#include "XT_Constants.h"
#include "allocate.h"
#include <math.h>
#include "XT_IOMisc.h"
#include "XT_AMatrix.h"
#include "XT_Profile.h"
#include "randlib.h"
#include "XT_Init.h"
#include "XT_Debug.h"
#include <fftw3.h>
#include "XT_CmplxArith.h"
#include "XT_FresnelTran.h"
#include "XT_MPIIO.h"

/*generates projection data from phantom*/
int32_t ForwardProject (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, int32_t proj_rows, int32_t proj_cols, float *measurements, float *brights)
{
	FILE *fp;
	long int stream_offset, size, result;
	int32_t i, j, k, m, n, idx, t, slice, r_subsmpl, t_subsmpl, r_idx, t_idx, r_origidx, t_origidx; 
	Real_t measurement_avg = 0, magpixel, phasepixel, val, expval, real, imag, measure_buf;
  	uint8_t AvgNumXElements, AvgNumZElements;
	char phantom_file[1000];
	int dimTiff[4];

	Real_arr_t* tifarray = (Real_arr_t*)get_spc(SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r, sizeof(Real_arr_t));
	float*** magobject = (float***)multialloc(sizeof(float), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	float*** phaseobject = (float***)multialloc(sizeof(float), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	Real_arr_t*** realmagobject = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	Real_arr_t*** realphaseobject = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x);
	
	Real_arr_t*** projs_real = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_t, SinogramPtr->N_r);
	Real_arr_t*** projs_imag = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_t, SinogramPtr->N_r);
	Real_arr_t*** fftforw_space = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_t, SinogramPtr->N_r);
	Real_arr_t*** fftback_space = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_t, SinogramPtr->N_r);
/*	Real_arr_t*** fftforw_freq = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_t, SinogramPtr->N_r);
	Real_arr_t*** fftback_freq = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, SinogramPtr->N_p, SinogramPtr->N_t, SinogramPtr->N_r);*/

	fftw_complex **fftforw_arr, **fftback_arr;
	fftw_plan *fftforw_plan, *fftback_plan;
    	fftforw_arr = (fftw_complex**)get_spc(SinogramPtr->N_p, sizeof(fftw_complex*));
    	fftback_arr = (fftw_complex**)get_spc(SinogramPtr->N_p, sizeof(fftw_complex*));
    	fftforw_plan = (fftw_plan*)get_spc(SinogramPtr->N_p, sizeof(fftw_plan));
    	fftback_plan = (fftw_plan*)get_spc(SinogramPtr->N_p, sizeof(fftw_plan));
	
	for (i = 0; i <  SinogramPtr->N_p; i++)
	{
		fftforw_arr[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*SinogramPtr->N_t*SinogramPtr->N_r);
		fftback_arr[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*SinogramPtr->N_t*SinogramPtr->N_r);
		fftforw_plan[i] = fftw_plan_dft_2d(SinogramPtr->N_r, SinogramPtr->N_t, fftforw_arr[i], fftforw_arr[i], FFTW_FORWARD, FFTW_ESTIMATE);
		fftback_plan[i] = fftw_plan_dft_2d(SinogramPtr->N_r, SinogramPtr->N_t, fftback_arr[i], fftback_arr[i], FFTW_BACKWARD, FFTW_ESTIMATE);
	}

	memset(&(projs_real[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));	  
	memset(&(projs_imag[0][0][0]), 0, SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r*sizeof(Real_arr_t));	    
	memset(&(measurements[0]), 0, SinogramPtr->N_p*proj_rows*proj_cols*sizeof(float));
	memset(&(brights[0]), 0, proj_cols*proj_rows*sizeof(float));

	/*AvgNumXElements over estimates the total number of entries in a single column of A matrix when indexed by both voxel and angle*/
  	AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/(SinogramPtr->delta_r));
	SinogramPtr->DetectorResponse = (Real_arr_t **)multialloc(sizeof(Real_arr_t), 2, SinogramPtr->N_p, DETECTOR_RESPONSE_BINS+1);
	SinogramPtr->ZLineResponse = (Real_arr_t *)get_spc(DETECTOR_RESPONSE_BINS + 1, sizeof(Real_arr_t));
	DetectorResponseProfile (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
	ZLineResponseProfile (SinogramPtr, ScannedObjectPtr, TomoInputsPtr);
	
  	AvgNumZElements = (uint8_t)((ScannedObjectPtr->delta_z/SinogramPtr->delta_t) + 2);
	
	AMatrixCol* VoxelLineResponse = (AMatrixCol*)get_spc(ScannedObjectPtr->N_z,sizeof(AMatrixCol));
	for (t = 0; t < ScannedObjectPtr->N_z; t++){
    		VoxelLineResponse[t].values = (Real_t*)get_spc(AvgNumZElements, sizeof(Real_t));
    		VoxelLineResponse[t].index = (int32_t*)get_spc(AvgNumZElements, sizeof(int32_t));
	}
	storeVoxelLineResponse(VoxelLineResponse, ScannedObjectPtr, SinogramPtr);

	r_subsmpl = PHANTOM_XY_SIZE/proj_cols;
	t_subsmpl = PHANTOM_Z_SIZE/proj_rows;

	sprintf(phantom_file, "%s", MAG_PHANTOM_FILEPATH);
	fp = fopen (phantom_file, "rb");
	check_error(fp==NULL, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Error in reading file %s\n", phantom_file);		
	size = (long int)ScannedObjectPtr->N_z*(long int)ScannedObjectPtr->N_y*(long int)ScannedObjectPtr->N_x;
	check_info(TomoInputsPtr->node_rank==0,TomoInputsPtr->debug_file_ptr, "Forward projecting mag phantom ...\n");	
/*	stream_offset = (long int)PHANTOM_OFFSET*(long int)ScannedObjectPtr->N_z*(long int)ScannedObjectPtr->N_y*(long int)ScannedObjectPtr->N_x*(long int)TomoInputsPtr->node_num;  */
	stream_offset = (long int)ScannedObjectPtr->N_z*(long int)ScannedObjectPtr->N_y*(long int)ScannedObjectPtr->N_x*(long int)TomoInputsPtr->node_rank;
	result = fseek (fp, stream_offset*sizeof(float), SEEK_SET);
  	check_error(result != 0, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_file,stream_offset);
	result = fread (&(magobject[0][0][0]), sizeof(float), size, fp);
  	check_error(result != size, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_file,result,stream_offset,size);
	fclose(fp);	
	
	sprintf(phantom_file, "%s", PHASE_PHANTOM_FILEPATH);
	fp = fopen (phantom_file, "rb");
	check_error(fp==NULL, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Error in reading file %s\n", phantom_file);		
	size = (long int)ScannedObjectPtr->N_z*(long int)ScannedObjectPtr->N_y*(long int)ScannedObjectPtr->N_x;
	check_info(TomoInputsPtr->node_rank==0,TomoInputsPtr->debug_file_ptr, "Forward projecting phase phantom ...\n");	
	stream_offset = (long int)ScannedObjectPtr->N_z*(long int)ScannedObjectPtr->N_y*(long int)ScannedObjectPtr->N_x*(long int)TomoInputsPtr->node_rank;
	result = fseek (fp, stream_offset*sizeof(float), SEEK_SET);
  	check_error(result != 0, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_file,stream_offset);
	result = fread (&(phaseobject[0][0][0]), sizeof(float), size, fp);
  	check_error(result != size, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_file,result,stream_offset,size);
	fclose(fp);	
	
  	#pragma omp parallel for private(i,j,k,slice,magpixel,phasepixel,idx,val,m,n)
	for (i=0; i<SinogramPtr->N_p; i++){
		AMatrixCol AMatrix;
  		AMatrix.values = (Real_t*)get_spc((int32_t)AvgNumXElements,sizeof(Real_t));
  		AMatrix.index  = (int32_t*)get_spc((int32_t)AvgNumXElements,sizeof(int32_t));

		for (j=0; j<ScannedObjectPtr->N_y; j++)
		for (k=0; k<ScannedObjectPtr->N_x; k++){	
	   	    	calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, SinogramPtr->DetectorResponse, &AMatrix, j, k, i, SinogramPtr->Light_Wavenumber); 
                	for (slice=0; slice<ScannedObjectPtr->N_z; slice++){
			    	magpixel = (Real_t)(magobject[slice][j][k]);
/*				if (magpixel < 0)
					magpixel = 0;
				else
					magpixel = (ABSORP_COEF_2 - ABSORP_COEF_1)*magpixel + ABSORP_COEF_1;*/
				realmagobject[slice][j][k] = (Real_arr_t)magpixel;
 
			    	phasepixel = (Real_t)(phaseobject[slice][j][k]);
/*				if (phasepixel < 0)
					phasepixel = 0;
				else
					phasepixel = (REF_IND_DEC_2 - REF_IND_DEC_1)*phasepixel + REF_IND_DEC_1;*/
				realphaseobject[slice][j][k] = (Real_arr_t)phasepixel;
				
			    	/*phasepixel = 0;*/
				/*IMPORTANT: Always make sure phantom has no negative values.*/
	     	          	for (m=0; m<AMatrix.count; m++){
                            		idx=AMatrix.index[m];
                            		val=AMatrix.values[m];
                            		for (n=0; n<VoxelLineResponse[slice].count; n++)
					{
                                    		projs_real[i][VoxelLineResponse[slice].index[n]][idx] += magpixel*val*VoxelLineResponse[slice].values[n];
                                    		projs_imag[i][VoxelLineResponse[slice].index[n]][idx] += phasepixel*val*VoxelLineResponse[slice].values[n];
	     				}
				}
			}
	  	 }

		free(AMatrix.values);
		free(AMatrix.index);
	}

	if (TomoInputsPtr->Write2Tiff == 1)
	{
		dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_t; dimTiff[3] = SinogramPtr->N_r;
		if (WriteMultiDimArray2Tiff ("SimMagProjs", dimTiff, 0, 1, 2, 3, &(projs_real[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
		if (WriteMultiDimArray2Tiff ("SimPhaseProjs", dimTiff, 0, 1, 2, 3, &(projs_imag[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
		dimTiff[0] = 1; dimTiff[1] = ScannedObjectPtr->N_z; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;
		if (WriteMultiDimArray2Tiff ("mag_phantom", dimTiff, 0, 1, 2, 3, &(realmagobject[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
		if (WriteMultiDimArray2Tiff ("phase_phantom", dimTiff, 0, 1, 2, 3, &(realphaseobject[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
	}
	size = SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r;	
	write_SharedBinFile_At ("SimMagProjs", &(projs_real[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	write_SharedBinFile_At ("SimPhaseProjs", &(projs_imag[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	
	size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
	write_SharedBinFile_At ("mag_phantom", &(realmagobject[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	write_SharedBinFile_At ("phase_phantom", &(realphaseobject[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	
	measurement_avg = 0;
  	
	for (slice=0; slice < proj_rows; slice++)
	for (j=0; j < proj_cols; j++)
		brights[slice*proj_cols + j] = EXPECTED_COUNT_MEASUREMENT; 
	
	check_info(TomoInputsPtr->node_rank==0,TomoInputsPtr->debug_file_ptr, "The expected count is %d\n", EXPECTED_COUNT_MEASUREMENT);	

/*  	#pragma omp parallel for private(slice, j, expval, real, imag, idx) reduction(+:measurement_avg)*/
	for (i=0; i < SinogramPtr->N_p; i++)
	{
		for (slice=0; slice < SinogramPtr->N_t; slice++)
		for (j=0; j < SinogramPtr->N_r; j++)
		{
			expval = exp(-projs_real[i][slice][j]);
			/*fftarr[j*SinogramPtr->N_t + slice][0] = EXPECTED_COUNT_MEASUREMENT*expval*cos(-projs_imag[i][slice][j]);
			fftarr[j*SinogramPtr->N_t + slice][1] = EXPECTED_COUNT_MEASUREMENT*expval*sin(-projs_imag[i][slice][j]);*/
			cmplx_mult (&real, &imag, expval*cos(-projs_imag[i][slice][j]), expval*sin(-projs_imag[i][slice][j]), sqrt(EXPECTED_COUNT_MEASUREMENT), (Real_t)(0));
			fftforw_arr[i][j*SinogramPtr->N_t + slice][0] = real;
			fftforw_arr[i][j*SinogramPtr->N_t + slice][1] = imag;
			/*weights[2*idx] = (val + sqrt(fabs(val))*normal());*/
			/*weights[2*idx+1] = (val + sqrt(fabs(val))*normal());*/
			fftforw_space[i][slice][j] = sqrt(real*real + imag*imag);
		}
		
/*		fftw_execute(fftforw_plan);
		for (slice=0; slice < SinogramPtr->N_t; slice++)
		for (j=0; j < SinogramPtr->N_r; j++)
			fftforw_freq[i][slice][j] = sqrt(fftforw_arr[j*SinogramPtr->N_t + slice][0]*fftforw_arr[j*SinogramPtr->N_t + slice][0] + fftforw_arr[j*SinogramPtr->N_t + slice][1]*fftforw_arr[j*SinogramPtr->N_t + slice][1]);*/
		compute_FresnelTran (SinogramPtr->N_r, SinogramPtr->N_t, SinogramPtr->delta_r, SinogramPtr->delta_t, fftforw_arr[i], &(fftforw_plan[i]), fftback_arr[i], &(fftback_plan[i]), SinogramPtr->Light_Wavelength, SinogramPtr->Obj2Det_Distance, SinogramPtr->Freq_Window);
/*		for (slice=0; slice < SinogramPtr->N_t; slice++)
		for (j=0; j < SinogramPtr->N_r; j++)
			fftback_freq[i][slice][j] = sqrt(fftback_arr[j*SinogramPtr->N_t
 + slice][0]*fftback_arr[j*SinogramPtr->N_t + slice][0] + fftback_arr[j*SinogramPtr->N_t + slice][1]*fftback_arr[j*SinogramPtr->N_t + slice][1]);
		fftw_execute(fftback_plan);*/

		for (slice=0; slice < proj_rows; slice++)
		for (j=0; j < proj_cols; j++)
		{
			idx = i*proj_rows*proj_cols + slice*proj_cols + j;
			for (t_idx = 0; t_idx < t_subsmpl; t_idx++)
			for (r_idx = 0; r_idx < r_subsmpl; r_idx++)
			{
				t_origidx = slice*t_subsmpl + t_idx;
				r_origidx = j*r_subsmpl + r_idx;
				
				measure_buf = (fftback_arr[i][r_origidx*SinogramPtr->N_t + t_origidx][0]*fftback_arr[i][r_origidx*SinogramPtr->N_t + t_origidx][0] + fftback_arr[i][r_origidx*SinogramPtr->N_t + t_origidx][1]*fftback_arr[i][r_origidx*SinogramPtr->N_t + t_origidx][1]);
				measurements[idx] += measure_buf;
			
				fftback_space[i][t_origidx][r_origidx] = sqrt(measure_buf);
			
			/*measurements[idx] = sqrt(fabs(measurements[idx]));*/
			/*weights[idx] = 1.0/measurements[idx];
			
			weight_avg += weights[idx];*/	
			}
			measurements[idx] /= (t_subsmpl*r_subsmpl);
			measurements[idx] = fabs(measurements[idx] + sqrt(fabs(measurements[idx]))*normal());
			measurement_avg += measurements[idx];	
		}
	}

	if (TomoInputsPtr->Write2Tiff == 1)
	{
		size = SinogramPtr->N_p*proj_rows*proj_cols;
		dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = proj_rows; dimTiff[3] = proj_cols;
/*		for (i = 0; i < size; i++) tifarray[i] = measurements[2*i];
		if (WriteMultiDimArray2Tiff ("measurements_real", dimTiff, 0, 2, 1, 3, &(tifarray[0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
		for (i = 0; i < size; i++) tifarray[i] = measurements[2*i+1];
		if (WriteMultiDimArray2Tiff ("measurements_imag", dimTiff, 0, 2, 1, 3, &(tifarray[0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}*/
		for (i = 0; i < size; i++) tifarray[i] = measurements[i];
		if (WriteMultiDimArray2Tiff ("measurements", dimTiff, 0, 1, 2, 3, &(tifarray[0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
    		write_SharedBinFile_At ("measurements", &(tifarray[0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
		
/*		for (i = 0; i < size; i++) tifarray[i] = weights[i];
		if (WriteMultiDimArray2Tiff ("weights", dimTiff, 0, 2, 1, 3, &(tifarray[0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
    		write_SharedBinFile_At ("weights", &(tifarray[0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);*/
		size = proj_rows*proj_cols;
		dimTiff[0] = 1; dimTiff[1] = 1; dimTiff[2] = proj_rows; dimTiff[3] = proj_cols;
		for (i = 0; i < size; i++) tifarray[i] = brights[i];
		if (WriteMultiDimArray2Tiff ("brights", dimTiff, 0, 1, 2, 3, &(tifarray[0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
    		write_SharedBinFile_At ("brights", &(tifarray[0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
		
		dimTiff[0] = 1; dimTiff[1] = SinogramPtr->N_p; dimTiff[2] = SinogramPtr->N_t; dimTiff[3] = SinogramPtr->N_r;
		size = SinogramPtr->N_p*SinogramPtr->N_t*SinogramPtr->N_r;
		if (WriteMultiDimArray2Tiff ("fftforw_space", dimTiff, 0, 1, 2, 3, &(fftforw_space[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
    		write_SharedBinFile_At ("fftforw_space", &(fftforw_space[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
		if (WriteMultiDimArray2Tiff ("fftback_space", dimTiff, 0, 1, 2, 3, &(fftback_space[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
    		write_SharedBinFile_At ("fftback_space", &(fftback_space[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
		/*if (WriteMultiDimArray2Tiff ("fftforw_freq", dimTiff, 0, 1, 2, 3, &(fftforw_freq[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
    		write_SharedBinFile_At ("fftforw_freq", &(fftforw_freq[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
		if (WriteMultiDimArray2Tiff ("fftback_freq", dimTiff, 0, 1, 2, 3, &(fftback_freq[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) {goto error;}
    		write_SharedBinFile_At ("fftback_freq", &(fftback_freq[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);*/
	}

	measurement_avg /= (SinogramPtr->N_p*proj_cols*proj_rows);
	printf("genSinogramFromPhantom: The average of all measurement data with/without noise is %f\n", measurement_avg);
	
        free(VoxelLineResponse->values);
        free(VoxelLineResponse->index);
	multifree(SinogramPtr->DetectorResponse,2);
	free(SinogramPtr->ZLineResponse);
        free(VoxelLineResponse);
	multifree(magobject,3);
	multifree(phaseobject,3);
	multifree(realmagobject,3);
	multifree(realphaseobject,3);
	multifree(projs_real,3);
	multifree(projs_imag,3);

	multifree(fftforw_space, 3); 
	multifree(fftback_space, 3);
	free(tifarray); 
    	for (i = 0; i < SinogramPtr->N_p; i++)
    	{ 
		fftw_destroy_plan(fftforw_plan[i]);
		fftw_destroy_plan(fftback_plan[i]);
        	fftw_free(fftforw_arr[i]); 
		fftw_free(fftback_arr[i]);
    	}
	free(fftforw_arr);
    	free(fftback_arr);
    
/*	multifree(fftforw_freq, 3); 
	multifree(fftback_freq, 3); */
	return (0);
error:
        free(VoxelLineResponse->values);
        free(VoxelLineResponse->index);
	multifree(SinogramPtr->DetectorResponse,2);
	free(SinogramPtr->ZLineResponse);
        free(VoxelLineResponse);
	multifree(magobject,3);
	multifree(phaseobject,3);
	multifree(projs_real,3);
	multifree(projs_imag,3);
	free(tifarray);
	return (-1);
}


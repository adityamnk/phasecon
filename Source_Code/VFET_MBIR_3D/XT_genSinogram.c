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
#include "XT_DensityUpdate.h"

/*generates projection data from phantom*/
int32_t ForwardProject (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, float *data_unflip_x, float* data_flip_x, float* data_unflip_y, float* data_flip_y)
{
	FILE *fp;
	long int stream_offset, size, result;
	int32_t i, j, k, m, n, idx, t, slice, r_subsmpl, t_subsmpl, data_idx; 
  	uint8_t AvgNumXElements, AvgNumZElements;
	char phantom_file[1000];
	int dimTiff[4];
	Real_t val, MagPhaseMultiple, ElecPhaseMultiple; Real_arr_t *objptr;

	MagPhaseMultiple = InpPtr->MagPhaseMultiple; 
	ElecPhaseMultiple = InpPtr->ElecPhaseMultiple; 

	Real_arr_t**** magobject = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
	Real_arr_t*** elecobject = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);
	
	Real_arr_t**** magpot = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
	Real_arr_t*** elecpot = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);

  	memset(data_unflip_x, 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(float));
  	memset(data_flip_x, 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(float));
  	memset(data_unflip_y, 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(float));
  	memset(data_flip_y, 0, SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r*sizeof(float));
/*	Real_arr_t*** realmagobject = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
	Real_arr_t*** realelecobject = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);
*/
	
	/*AvgNumXElements over estimates the total number of entries in a single column of A matrix when indexed by both voxel and angle*/
  	AvgNumXElements = (uint8_t)ceil(3*ObjPtr->delta_xy/(SinoPtr->delta_r) + 2);
	SinoPtr->DetectorResponse = (Real_arr_t **)multialloc(sizeof(Real_arr_t), 2, SinoPtr->N_p, DETECTOR_RESPONSE_BINS+1);
	SinoPtr->ZLineResponse = (Real_arr_t *)get_spc(DETECTOR_RESPONSE_BINS + 1, sizeof(Real_arr_t));
	DetectorResponseProfile (SinoPtr, ObjPtr, InpPtr);
	ZLineResponseProfile (SinoPtr, ObjPtr, InpPtr);
	
  	AvgNumZElements = (uint8_t)((ObjPtr->delta_z/SinoPtr->delta_t) + 2);
	
	AMatrixCol* VoxelLineResponse = (AMatrixCol*)get_spc(ObjPtr->N_z,sizeof(AMatrixCol));
	for (t = 0; t < ObjPtr->N_z; t++){
    		VoxelLineResponse[t].values = (Real_t*)get_spc(AvgNumZElements, sizeof(Real_t));
    		VoxelLineResponse[t].index = (int32_t*)get_spc(AvgNumZElements, sizeof(int32_t));
	}
	storeVoxelLineResponse(VoxelLineResponse, ObjPtr, SinoPtr);

	r_subsmpl = ObjPtr->N_x/SinoPtr->N_r;
	t_subsmpl = ObjPtr->N_z/SinoPtr->N_t;

	sprintf(phantom_file, "%s.bin", PHANTOM_MAGDENSITY_FILENAME);
	fp = fopen (phantom_file, "rb");
	check_error(fp==NULL, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Error in reading file %s\n", phantom_file);		
	size = (long int)ObjPtr->N_z*(long int)ObjPtr->N_y*(long int)ObjPtr->N_x*3;
	check_info(InpPtr->node_rank==0,InpPtr->debug_file_ptr, "Forward projecting mag phantom ...\n");	
/*	stream_offset = (long int)PHANTOM_OFFSET*(long int)ObjPtr->N_z*(long int)ObjPtr->N_y*(long int)ObjPtr->N_x*(long int)InpPtr->node_num;  */
	stream_offset = (long int)ObjPtr->N_z*(long int)ObjPtr->N_y*(long int)ObjPtr->N_x*(long int)InpPtr->node_rank;
	result = fseek (fp, stream_offset*sizeof(Real_arr_t), SEEK_SET);
  	check_error(result != 0, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_file,stream_offset);
	result = fread (&(magobject[0][0][0][0]), sizeof(Real_arr_t), size, fp);
  	check_error(result != size, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_file,result,stream_offset,size);
	fclose(fp);	
	
	sprintf(phantom_file, "%s.bin", PHANTOM_ELECDENSITY_FILENAME);
	fp = fopen (phantom_file, "rb");
	check_error(fp==NULL, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Error in reading file %s\n", phantom_file);		
	size = (long int)ObjPtr->N_z*(long int)ObjPtr->N_y*(long int)ObjPtr->N_x;
	check_info(InpPtr->node_rank==0,InpPtr->debug_file_ptr, "Forward projecting elec phantom ...\n");	
	stream_offset = (long int)ObjPtr->N_z*(long int)ObjPtr->N_y*(long int)ObjPtr->N_x*(long int)InpPtr->node_rank;
	result = fseek (fp, stream_offset*sizeof(Real_arr_t), SEEK_SET);
  	check_error(result != 0, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_file,stream_offset);
	result = fread (&(elecobject[0][0][0]), sizeof(Real_arr_t), size, fp);
  	check_error(result != size, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_file,result,stream_offset,size);
	fclose(fp);	

  	compute_magcrossprodtran (magobject, magpot, ObjPtr->MagFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 1);
  	compute_elecprodtran (elecobject, elecpot, ObjPtr->ElecFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 1);

	Write2Bin (PHANTOM_MAGDENSITY_FILENAME, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3, sizeof(Real_arr_t), &(magobject[0][0][0][0]), InpPtr->debug_file_ptr);
	Write2Bin (PHANTOM_ELECDENSITY_FILENAME, 1, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, sizeof(Real_arr_t), &(elecobject[0][0][0]), InpPtr->debug_file_ptr);
	Write2Bin (PHANTOM_MAGVECPOT_FILENAME, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3, sizeof(Real_arr_t), &(magpot[0][0][0][0]), InpPtr->debug_file_ptr);
	Write2Bin (PHANTOM_ELECPOT_FILENAME, 1, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, sizeof(Real_arr_t), &(elecpot[0][0][0]), InpPtr->debug_file_ptr);
	
	/*fp = fopen (phantom_file, "rb");
	check_error(fp==NULL, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "Error in reading file %s\n", phantom_file);		
	size = (long int)ObjPtr->N_z*(long int)ObjPtr->N_y*(long int)ObjPtr->N_x;
	check_info(InpPtr->node_rank==0,InpPtr->debug_file_ptr, "Forward projecting elec phantom ...\n");	
	stream_offset = (long int)ObjPtr->N_z*(long int)ObjPtr->N_y*(long int)ObjPtr->N_x*(long int)InpPtr->node_rank;
	result = fseek (fp, stream_offset*sizeof(Real_arr_t), SEEK_SET);
  	check_error(result != 0, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_file,stream_offset);
	result = fread (&(elecpot[0][0][0]), sizeof(Real_arr_t), size, fp);
  	check_error(result != size, InpPtr->node_rank==0, InpPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_file,result,stream_offset,size);
	fclose(fp);*/	
	
  	#pragma omp parallel for private(i,j,k,slice,idx,val,m,n,data_idx)
	for (i=0; i<SinoPtr->N_p; i++){
		AMatrixCol AMatrix;
  		AMatrix.values = (Real_t*)get_spc((int32_t)AvgNumXElements,sizeof(Real_t));
  		AMatrix.index  = (int32_t*)get_spc((int32_t)AvgNumXElements,sizeof(int32_t));

		for (j=0; j<ObjPtr->N_z; j++)
		for (k=0; k<ObjPtr->N_y; k++){	
	   	    	calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse, &AMatrix, j, k, i); 
                	for (slice=0; slice<ObjPtr->N_x; slice++){
	     	          	for (m=0; m<AMatrix.count; m++){
                            		idx=AMatrix.index[m];
                            		val=AMatrix.values[m];
                            		for (n=0; n<VoxelLineResponse[slice].count; n++)
					{
						data_idx = i*SinoPtr->N_t*SinoPtr->N_r + idx*SinoPtr->N_t + VoxelLineResponse[slice].index[n];

                                    		data_unflip_x[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*magpot[j][k][slice][0]*SinoPtr->cosine[i];
                                    		data_unflip_x[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*magpot[j][k][slice][1]*(-SinoPtr->sine[i]);
                                    		data_unflip_x[data_idx] += val*ElecPhaseMultiple*VoxelLineResponse[slice].values[n]*elecpot[j][k][slice];
                                    		
                                    		data_flip_x[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*(-magpot[j][k][slice][0])*SinoPtr->cosine[i];
						data_flip_x[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*(-magpot[j][k][slice][1])*(-SinoPtr->sine[i]);
                                    		data_flip_x[data_idx] += val*ElecPhaseMultiple*VoxelLineResponse[slice].values[n]*elecpot[j][k][slice];
	     				}
				}
			}
	  	 }
		
		for (j=0; j<ObjPtr->N_z; j++)
		for (k=0; k<ObjPtr->N_x; k++){	
	   	    	calcAMatrixColumnforAngle(SinoPtr, ObjPtr, SinoPtr->DetectorResponse, &AMatrix, j, k, i); 
                	for (slice=0; slice<ObjPtr->N_y; slice++){
	     	          	for (m=0; m<AMatrix.count; m++){
                            		idx=AMatrix.index[m];
                            		val=AMatrix.values[m];
                            		for (n=0; n<VoxelLineResponse[slice].count; n++)
					{
						data_idx = i*SinoPtr->N_t*SinoPtr->N_r + VoxelLineResponse[slice].index[n]*SinoPtr->N_r + idx;
                                    	
                                    		data_unflip_y[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*magpot[j][slice][k][0]*SinoPtr->cosine[i];
						data_unflip_y[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*magpot[j][slice][k][2]*(-SinoPtr->sine[i]);
                                    		data_unflip_y[data_idx] += val*ElecPhaseMultiple*VoxelLineResponse[slice].values[n]*elecpot[j][slice][k];
                                    		
                                    		data_flip_y[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*(-magpot[j][slice][k][0])*SinoPtr->cosine[i];
						data_flip_y[data_idx] += val*MagPhaseMultiple*VoxelLineResponse[slice].values[n]*(-magpot[j][slice][k][2])*(-SinoPtr->sine[i]);
                                    		data_flip_y[data_idx] += val*ElecPhaseMultiple*VoxelLineResponse[slice].values[n]*elecpot[j][slice][k];
	     				}
				}
			}
	  	 }

		free(AMatrix.values);
		free(AMatrix.index);
	}

	if (InpPtr->Write2Tiff == 1)
	{
		Real_arr_t* tifarray = (Real_arr_t*)get_spc(SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r, sizeof(Real_arr_t));
		
		size = SinoPtr->N_p*SinoPtr->N_t*SinoPtr->N_r;
		dimTiff[0] = 1; dimTiff[1] = SinoPtr->N_p; dimTiff[2] = SinoPtr->N_r; dimTiff[3] = SinoPtr->N_t;

		for (i = 0; i < size; i++) tifarray[i] = data_unflip_x[i];
		if (WriteMultiDimArray2Tiff ("sim_data_unflip_x", dimTiff, 0, 1, 2, 3, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}

		for (i = 0; i < size; i++) tifarray[i] = data_flip_x[i];
		if (WriteMultiDimArray2Tiff ("sim_data_flip_x", dimTiff, 0, 1, 2, 3, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}
		
		dimTiff[0] = 1; dimTiff[1] = SinoPtr->N_p; dimTiff[2] = SinoPtr->N_t; dimTiff[3] = SinoPtr->N_r;
		for (i = 0; i < size; i++) tifarray[i] = data_unflip_y[i];
		if (WriteMultiDimArray2Tiff ("sim_data_unflip_y", dimTiff, 0, 1, 2, 3, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}
		
		for (i = 0; i < size; i++) tifarray[i] = data_flip_y[i];
		if (WriteMultiDimArray2Tiff ("sim_data_flip_y", dimTiff, 0, 1, 2, 3, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}

		free(tifarray); 
		tifarray = (Real_arr_t*)get_spc(ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*3, sizeof(Real_arr_t));	

		size = ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*3;	
		dimTiff[0] = ObjPtr->N_z; dimTiff[1] = ObjPtr->N_y; dimTiff[2] = ObjPtr->N_x; dimTiff[3] = 3;

		objptr = &(magobject[0][0][0][0]);
		for (i = 0; i < size; i++) tifarray[i] = objptr[i]; 
		if (WriteMultiDimArray2Tiff (PHANTOM_MAGDENSITY_FILENAME, dimTiff, 0, 3, 1, 2, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}
		
		objptr = &(magpot[0][0][0][0]);
		for (i = 0; i < size; i++) tifarray[i] = objptr[i]; 
		if (WriteMultiDimArray2Tiff (PHANTOM_MAGVECPOT_FILENAME, dimTiff, 0, 3, 1, 2, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}
		
		size = ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x;	
		dimTiff[0] = 1; dimTiff[1] = ObjPtr->N_z; dimTiff[2] = ObjPtr->N_y; dimTiff[3] = ObjPtr->N_x;

		objptr = &(elecobject[0][0][0]);
		for (i = 0; i < size; i++) tifarray[i] = objptr[i]; 
		if (WriteMultiDimArray2Tiff (PHANTOM_ELECDENSITY_FILENAME, dimTiff, 0, 1, 2, 3, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}
		
		objptr = &(elecpot[0][0][0]);
		for (i = 0; i < size; i++) tifarray[i] = objptr[i]; 
		if (WriteMultiDimArray2Tiff (PHANTOM_ELECPOT_FILENAME, dimTiff, 0, 1, 2, 3, tifarray, 0, 0, 1, InpPtr->debug_file_ptr)) {goto error;}
		
		free(tifarray); 
	}
	
/*	size = ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x;
	write_SharedBinFile_At ("mag_phantom", &(realmagobject[0][0][0]), InpPtr->node_rank*size, size, InpPtr->debug_file_ptr);
	write_SharedBinFile_At ("phase_phantom", &(realphaseobject[0][0][0]), InpPtr->node_rank*size, size, InpPtr->debug_file_ptr);*/
	
	for (t = 0; t < ObjPtr->N_z; t++){
        	free(VoxelLineResponse[t].values);
        	free(VoxelLineResponse[t].index);
	}
	multifree(SinoPtr->DetectorResponse,2);
	free(SinoPtr->ZLineResponse);
        free(VoxelLineResponse);
	multifree(magobject,4);
	multifree(elecobject,3);
	multifree(magpot,4);
	multifree(elecpot,3);
	/*multifree(realmagobject,3);
	multifree(realphaseobject,3);*/

    
/*	multifree(fftforw_freq, 3); 
	multifree(fftback_freq, 3); */
	return (0);
error:
	for (t = 0; t < ObjPtr->N_z; t++){
        	free(VoxelLineResponse[t].values);
        	free(VoxelLineResponse[t].index);
	}
	multifree(SinoPtr->DetectorResponse,2);
	free(SinoPtr->ZLineResponse);
        free(VoxelLineResponse);
	multifree(magobject,4);
	multifree(elecobject,3);
	return (-1);
}


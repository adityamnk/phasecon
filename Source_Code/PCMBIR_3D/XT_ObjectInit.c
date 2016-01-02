#include <stdio.h>
#include <fftw3.h>
#include "XT_Constants.h"
#include "XT_Debug.h"
#include "XT_IOMisc.h"
#include "XT_AMatrix.h"
#include "XT_Structures.h"
#include "XT_MPIIO.h"
#include "allocate.h"
#include "XT_ForwardProject.h"
#include "XT_PhaseRet.h"
#include <math.h>

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


void gen_data_GroundTruth (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	float ***Init, *temparr;
	char object_file[100];
	int32_t dimTiff[4];
	Real_arr_t ***RealObj, ***ImagObj, ***RealSino, ***ImagSino;
	Real_t pixel; FILE *fp;
	int32_t N_z, N_y, N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, sino_idx, i, j, k, l, p, slice;
	long int size, result;
	char mag_phantom_filename[] = MAG_PHANTOM_FILEPATH;
	char phase_phantom_filename[] = PHASE_PHANTOM_FILEPATH;
	/*char phantom_filename[] = MAX_OBJ_FILEPATH;*/
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
	
	fp = fopen (mag_phantom_filename, "rb");
/*	check_error(fp==NULL, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Error in reading file %s\n", phantom_filename);*/
/*	stream_offset = (long int)PHANTOM_OFFSET * (long int)N_z * (long int)N_y * (long int)N_x;
	result = fseek (fp, stream_offset*sizeof(float), SEEK_SET);*/
/*  	check_error(result != 0, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_filename,stream_offset);*/
	result = fread (&(Init[0][0][0]), sizeof(float), size, fp);
/*  	check_error(result != size, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_filename,result,stream_offset,size);*/
	temparr = &(Init[0][0][0]);
/*  	#pragma omp parallel for
      	for (k=0; k<size; k++)
	{
		if (temparr[k] < 0) temparr[k] = 0;
		else temparr[k] = (ABSORP_COEF_2 - ABSORP_COEF_1)*temparr[k] + ABSORP_COEF_1; 
 	}*/
	dwnsmpl_object (RealObj, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 2);
	fclose(fp);
	
	fp = fopen (phase_phantom_filename, "rb");
/*	check_error(fp==NULL, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Error in reading file %s\n", phantom_filename);	*/	     
/*	stream_offset = (long int)PHANTOM_OFFSET * (long int)N_z * (long int)N_y * (long int)N_x;
	result = fseek (fp, stream_offset*sizeof(float), SEEK_SET);*/
/*  	check_error(result != 0, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Error in seeking file %s, stream_offset = %ld\n",phantom_filename,stream_offset);*/
	result = fread (&(Init[0][0][0]), sizeof(float), size, fp);
/*  	check_error(result != size, TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: Reading file %s, Number of elements read does not match required, number of elements read=%ld, stream_offset=%ld, size=%ld\n",phantom_filename,result,stream_offset,size);*/
	temparr = &(Init[0][0][0]);
 /* 	#pragma omp parallel for
      	for (k=0; k<size; k++)
	{
		if (temparr[k] < 0) temparr[k] = 0;
		else temparr[k] = (REF_IND_DEC_2 - REF_IND_DEC_1)*temparr[k] + REF_IND_DEC_1;
 	}*/
	dwnsmpl_object (ImagObj, Init, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, dwnsmpl_z, dwnsmpl_y, dwnsmpl_x, 2);
	fclose(fp);

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
          				calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, SinogramPtr->DetectorResponse, &(AMatrixPtr[i]), j, k, sino_idx, SinogramPtr->Light_Wavenumber);
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
		SinogramPtr->Omega_real[i][j][k] = RealSino[i][j][k];
		SinogramPtr->Omega_imag[i][j][k] = ImagSino[i][j][k];
	}
	
	size = SinogramPtr->N_p*SinogramPtr->N_r*SinogramPtr->N_t;
	write_SharedBinFile_At (PAG_MAGRET_FILENAME, &(SinogramPtr->Omega_real[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	write_SharedBinFile_At (PAG_PHASERET_FILENAME, &(SinogramPtr->Omega_imag[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	
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
			sprintf (object_file, "%s_n%d", PHANTOM_MAGOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->MagObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
			sprintf (object_file, "%s_n%d", PHANTOM_PHASEOBJECT_FILENAME, TomoInputsPtr->node_rank);
		    	sprintf (object_file, "%s_time_%d", object_file, i);
    			WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->PhaseObject[i][1][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr);
	  	}
	  
	size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;	
       	sprintf(object_file, "%s_time_%d", MAGOBJECT_FILENAME, 0);
	write_SharedBinFile_At (object_file, &(ScannedObjectPtr->MagObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
       	sprintf(object_file, "%s_time_%d", PHASEOBJECT_FILENAME, 0);
	write_SharedBinFile_At (object_file, &(ScannedObjectPtr->PhaseObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
       	
	write_SharedBinFile_At (PHANTOM_MAGOBJECT_FILENAME, &(ScannedObjectPtr->MagObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);
	write_SharedBinFile_At (PHANTOM_PHASEOBJECT_FILENAME, &(ScannedObjectPtr->PhaseObject[0][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr);

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
	/*if (TomoInputsPtr->recon_type == 2)
	{
		printf("Reading pag mag object file = %s\n", PAG_MAGOBJECT_FILENAME);
		if (read_SharedBinFile_At (PAG_MAGOBJECT_FILENAME, &(ScannedObjectPtr->MagObject[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
		printf("Reading pag phase object\n");
		if (read_SharedBinFile_At (PAG_PHASEOBJECT_FILENAME, &(ScannedObjectPtr->PhaseObject[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
      	}
	else
	{*/
		for (i = 0; i < ScannedObjectPtr->N_time; i++)
      		{
        		sprintf(object_file, "%s_time_%d", MAGOBJECT_FILENAME,i);
			if (read_SharedBinFile_At (object_file, &(ScannedObjectPtr->MagObject[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
        		sprintf(object_file, "%s_time_%d", PHASEOBJECT_FILENAME,i);
			if (read_SharedBinFile_At (object_file, &(ScannedObjectPtr->PhaseObject[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
/*		for (j = 0; j < ScannedObjectPtr->N_z; j++)
		for (k = 0; k < ScannedObjectPtr->N_y; k++)
		for (l = 0; l < ScannedObjectPtr->N_x; l++)
			if (ScannedObjectPtr->PhaseObject[i][j][k][l] > REF_IND_DEC_1/2.0)
				ScannedObjectPtr->MagObject[i][j][k][l] = (ABSORP_COEF_1 + ABSORP_COEF_2)/2.0;*/ 
      		}
	/*}*/
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


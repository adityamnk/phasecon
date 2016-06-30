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
void upsample_bilinear_2D (Real_arr_t*** Object, Real_arr_t*** Init, int32_t N_z, int32_t N_y, int32_t N_x)
{
  int32_t j, k, m;
  Real_arr_t **buffer;
  
  #pragma omp parallel for private(buffer, m, j, k)
  for (m=0; m < N_z; m++)
  {
    buffer = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, N_y, 2*N_x);
    for (j=0; j < N_y; j++){
      buffer[j][0] = Init[m][j][0];
      buffer[j][1] = (3.0*Init[m][j][0] + Init[m][j][1])/4.0;
      buffer[j][2*N_x - 1] = Init[m][j][N_x - 1];
      buffer[j][2*N_x - 2] = (Init[m][j][N_x - 2] + 3.0*Init[m][j][N_x - 1])/4.0;
      for (k=1; k < N_x - 1; k++){
        buffer[j][2*k] = (Init[m][j][k-1] + 3.0*Init[m][j][k])/4.0;
        buffer[j][2*k + 1] = (3.0*Init[m][j][k] + Init[m][j][k+1])/4.0;
      }
    }
    for (k=0; k < 2*N_x; k++){
      Object[m][0][k] = buffer[0][k];
      Object[m][1][k] = (3.0*buffer[0][k] + buffer[1][k])/4.0;
      Object[m][2*N_y-1][k] = buffer[N_y-1][k];
      Object[m][2*N_y-2][k] = (buffer[N_y-2][k] + 3.0*buffer[N_y-1][k])/4.0;
    }
    for (j=1; j<N_y-1; j++){
      for (k=0; k<2*N_x; k++){
        Object[m][2*j][k] = (buffer[j-1][k] + 3.0*buffer[j][k])/4.0;
        Object[m][2*j + 1][k] = (3*buffer[j][k] + buffer[j+1][k])/4.0;
      }
    }
    multifree(buffer,2);
  }
}

/*Upsamples the (N_z x N_y x N_x) size 'Init' by a factor of 2 along the x-y plane and stores it in 'Object'*/
void upsample_object_bilinear_2D (Real_arr_t**** MagPotentials, Real_arr_t*** ElecPotentials, Real_arr_t**** MagInit, Real_arr_t*** ElecInit, int32_t N_z, int32_t N_y, int32_t N_x)
{
  int32_t i, j, k, slice;
  Real_arr_t **buffer;
  
  buffer = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, N_y, 2*N_x);
  for (i = 0; i < 3; i++)
  {
  	for (slice=0; slice < N_z; slice++){
    	for (j=0; j < N_y; j++){
      		buffer[j][0] = MagInit[slice][j][0][i];
      		buffer[j][1] = (3.0*MagInit[slice][j][0][i] + MagInit[slice][j][1][i])/4.0;
      		buffer[j][2*N_x - 1] = MagInit[slice][j][N_x - 1][i];
      		buffer[j][2*N_x - 2] = (MagInit[slice][j][N_x - 2][i] + 3.0*MagInit[slice][j][N_x - 1][i])/4.0;
      		for (k=1; k < N_x - 1; k++){
        		buffer[j][2*k] = (MagInit[slice][j][k-1][i] + 3.0*MagInit[slice][j][k][i])/4.0;
        		buffer[j][2*k + 1] = (3.0*MagInit[slice][j][k][i] + MagInit[slice][j][k+1][i])/4.0;
      		}
    	}

    	for (k=0; k < 2*N_x; k++){
     		MagPotentials[slice+1][0][k][i] = buffer[0][k];
      		MagPotentials[slice+1][1][k][i] = (3.0*buffer[0][k] + buffer[1][k])/4.0;
      		MagPotentials[slice+1][2*N_y-1][k][i] = buffer[N_y-1][k];
      		MagPotentials[slice+1][2*N_y-2][k][i] = (buffer[N_y-2][k] + 3.0*buffer[N_y-1][k])/4.0;
    	}
    
	for (j=1; j<N_y-1; j++){
      	for (k=0; k<2*N_x; k++){
        	MagPotentials[slice+1][2*j][k][i] = (buffer[j-1][k] + 3.0*buffer[j][k])/4.0;
        	MagPotentials[slice+1][2*j + 1][k][i] = (3*buffer[j][k] + buffer[j+1][k])/4.0;
      	}
    	}
  	}
 }
 
#ifdef VFET_ELEC_RECON 
  for (slice=0; slice < N_z; slice++){
    for (j=0; j < N_y; j++){
      buffer[j][0] = ElecInit[slice][j][0];
      buffer[j][1] = (3.0*ElecInit[slice][j][0] + ElecInit[slice][j][1])/4.0;
      buffer[j][2*N_x - 1] = ElecInit[slice][j][N_x - 1];
      buffer[j][2*N_x - 2] = (ElecInit[slice][j][N_x - 2] + 3.0*ElecInit[slice][j][N_x - 1])/4.0;
      for (k=1; k < N_x - 1; k++){
        buffer[j][2*k] = (ElecInit[slice][j][k-1] + 3.0*ElecInit[slice][j][k])/4.0;
        buffer[j][2*k + 1] = (3.0*ElecInit[slice][j][k] + ElecInit[slice][j][k+1])/4.0;
      }
    }
    for (k=0; k < 2*N_x; k++){
      ElecPotentials[slice+1][0][k] = buffer[0][k];
      ElecPotentials[slice+1][1][k] = (3.0*buffer[0][k] + buffer[1][k])/4.0;
      ElecPotentials[slice+1][2*N_y-1][k] = buffer[N_y-1][k];
      ElecPotentials[slice+1][2*N_y-2][k] = (buffer[N_y-2][k] + 3.0*buffer[N_y-1][k])/4.0;
    }
    for (j=1; j<N_y-1; j++){
      for (k=0; k<2*N_x; k++){
        ElecPotentials[slice+1][2*j][k] = (buffer[j-1][k] + 3.0*buffer[j][k])/4.0;
        ElecPotentials[slice+1][2*j + 1][k] = (3*buffer[j][k] + buffer[j+1][k])/4.0;
      }
    }
  }
#endif
  multifree(buffer,2);
}

void upsample_bilinear_3D (Real_arr_t*** Object, Real_arr_t*** Init, int32_t N_z, int32_t N_y, int32_t N_x)
{
  int32_t j, k, slice;
  Real_t ***buffer2D, ***buffer3D;
	  
  buffer2D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, N_y, 2*N_x);
  buffer3D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, 2*N_y, 2*N_x);
  #pragma omp parallel for private(slice, j, k)
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
    		Object[0][j][k] = buffer3D[0][j][k];
    		Object[1][j][k] = (3.0*buffer3D[0][j][k] + buffer3D[1][j][k])/4.0;
    		Object[2*N_z-1][j][k] = buffer3D[N_z-1][j][k];
    		Object[2*N_z-2][j][k] = (3.0*buffer3D[N_z-1][j][k] + buffer3D[N_z-2][j][k])/4.0;
  	}
  
  	for (slice=1; slice < N_z-1; slice++)
  	for (j=0; j<2*N_y; j++)
  	for (k=0; k<2*N_x; k++){
    		Object[2*slice][j][k] = (buffer3D[slice-1][j][k] + 3.0*buffer3D[slice][j][k])/4.0;
    		Object[2*slice+1][j][k] = (3.0*buffer3D[slice][j][k] + buffer3D[slice+1][j][k])/4.0;
  	}
  
  	multifree(buffer2D,3);
  	multifree(buffer3D,3);
}

/*'InitObject' intializes the Object to be reconstructed to either 0 or an interpolated version of the previous reconstruction. It is used in multi resolution reconstruction in which after every coarse resolution reconstruction the object should be intialized with an interpolated version of the reconstruction following which the object will be reconstructed at a finer resolution.*/
/*Upsamples the (N_time x N_z x N_y x N_x) size 'Init' by a factor of 2 along the in 3D x-y-z coordinates and stores it in 'Object'*/
void upsample_object_bilinear_3D (Real_arr_t**** MagPotentials, Real_arr_t*** ElecPotentials, Real_arr_t**** MagInit, Real_arr_t*** ElecInit, int32_t N_z, int32_t N_y, int32_t N_x, int32_t z_off)
{
  int32_t i, j, k, slice;
  Real_t ***buffer2D, ***buffer3D;
  
  buffer2D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, N_y, 2*N_x);
  buffer3D = (Real_t***)multialloc(sizeof(Real_t), 3, N_z, 2*N_y, 2*N_x);
 
  for (i = 0; i < 3; i++)
  { 
  	for (slice=0; slice < N_z; slice++){
    		for (j=0; j < N_y; j++){
      			buffer2D[slice][j][0] = MagInit[slice][j][0][i];
      			buffer2D[slice][j][1] = (3.0*MagInit[slice][j][0][i] + MagInit[slice][j][1][i])/4.0;
      			buffer2D[slice][j][2*N_x - 1] = MagInit[slice][j][N_x - 1][i];
      			buffer2D[slice][j][2*N_x - 2] = (MagInit[slice][j][N_x - 2][i] + 3.0*MagInit[slice][j][N_x - 1][i])/4.0;
      			for (k=1; k < N_x - 1; k++){
       				buffer2D[slice][j][2*k] = (MagInit[slice][j][k-1][i] + 3.0*MagInit[slice][j][k][i])/4.0;
        			buffer2D[slice][j][2*k + 1] = (3.0*MagInit[slice][j][k][i] + MagInit[slice][j][k+1][i])/4.0;
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
  		MagPotentials[z_off][j][k][i] = buffer3D[0][j][k];
    		MagPotentials[1+z_off][j][k][i] = (3.0*buffer3D[0][j][k] + buffer3D[1][j][k])/4.0;
    		MagPotentials[2*N_z-1+z_off][j][k][i] = buffer3D[N_z-1][j][k];
    		MagPotentials[2*N_z-2+z_off][j][k][i] = (3.0*buffer3D[N_z-1][j][k] + buffer3D[N_z-2][j][k])/4.0;
  	}
  
  	for (slice=1; slice < N_z-1; slice++)
  	for (j=0; j<2*N_y; j++)
  	for (k=0; k<2*N_x; k++){
    		MagPotentials[2*slice+z_off][j][k][i] = (buffer3D[slice-1][j][k] + 3.0*buffer3D[slice][j][k])/4.0;
    		MagPotentials[2*slice+1+z_off][j][k][i] = (3.0*buffer3D[slice][j][k] + buffer3D[slice+1][j][k])/4.0;
  	}
  }
  
#ifdef VFET_ELEC_RECON 
  for (slice=0; slice < N_z; slice++){
    for (j=0; j < N_y; j++){
      buffer2D[slice][j][0] = ElecInit[slice][j][0];
      buffer2D[slice][j][1] = (3.0*ElecInit[slice][j][0] + ElecInit[slice][j][1])/4.0;
      buffer2D[slice][j][2*N_x - 1] = ElecInit[slice][j][N_x - 1];
      buffer2D[slice][j][2*N_x - 2] = (ElecInit[slice][j][N_x - 2] + 3.0*ElecInit[slice][j][N_x - 1])/4.0;
      for (k=1; k < N_x - 1; k++){
        buffer2D[slice][j][2*k] = (ElecInit[slice][j][k-1] + 3.0*ElecInit[slice][j][k])/4.0;
        buffer2D[slice][j][2*k + 1] = (3.0*ElecInit[slice][j][k] + ElecInit[slice][j][k+1])/4.0;
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
    ElecPotentials[z_off][j][k] = buffer3D[0][j][k];
    ElecPotentials[1+z_off][j][k] = (3.0*buffer3D[0][j][k] + buffer3D[1][j][k])/4.0;
    ElecPotentials[2*N_z-1+z_off][j][k] = buffer3D[N_z-1][j][k];
    ElecPotentials[2*N_z-2+z_off][j][k] = (3.0*buffer3D[N_z-1][j][k] + buffer3D[N_z-2][j][k])/4.0;
  }
  
  for (slice=1; slice < N_z-1; slice++)
  for (j=0; j<2*N_y; j++)
  for (k=0; k<2*N_x; k++){
    ElecPotentials[2*slice+z_off][j][k] = (buffer3D[slice-1][j][k] + 3.0*buffer3D[slice][j][k])/4.0;
    ElecPotentials[2*slice+1+z_off][j][k] = (3.0*buffer3D[slice][j][k] + buffer3D[slice+1][j][k])/4.0;
  }
#endif
  
  multifree(buffer2D,3);
  multifree(buffer3D,3);
}

void dwnsmpl_init_phantom (Real_arr_t**** MagObject, Real_t*** ElecObject, Real_t**** MagInit, Real_t*** ElecInit, int32_t N_z, int32_t N_y, int32_t N_x, int32_t dwnsmpl_z, int32_t dwnsmpl_y, int32_t dwnsmpl_x)
{
	int32_t i, j, k, m, n, p;
	
	for (i = 0; i < N_z; i++)
	for (j = 0; j < N_y; j++)
	for (k = 0; k < N_x; k++)
	{
		MagObject[i+1][j][k][0] = 0;
		MagObject[i+1][j][k][1] = 0;
		ElecObject[i+1][j][k] = 0;
		for (m = 0; m < dwnsmpl_z; m++)
		for (n = 0; n < dwnsmpl_y; n++)
		for (p = 0; p < dwnsmpl_x; p++)
		{
			MagObject[i+1][j][k][0] += MagInit[k*dwnsmpl_x + p][i*dwnsmpl_z + m][j*dwnsmpl_y + n][3];
			MagObject[i+1][j][k][1] += MagInit[k*dwnsmpl_x + p][i*dwnsmpl_z + m][j*dwnsmpl_y + n][2];
			ElecObject[i+1][j][k] += ElecInit[k*dwnsmpl_x + p][i*dwnsmpl_z + m][j*dwnsmpl_y + n];
		}
		
		MagObject[i+1][j][k][0] /= (dwnsmpl_z*dwnsmpl_y*dwnsmpl_x);
		MagObject[i+1][j][k][1] /= (dwnsmpl_z*dwnsmpl_y*dwnsmpl_x);
		ElecObject[i+1][j][k] /= (dwnsmpl_z*dwnsmpl_y*dwnsmpl_x);
	}
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
  int32_t j, k, l, flag = 0;
  Real_arr_t ****MagInit, ***ElecInit /*, ***UpMapInit*/;
  
  for (j = 0; j < ScannedObjectPtr->N_z; j++)
  for (k = 0; k < ScannedObjectPtr->N_y; k++)
  for (l = 0; l < ScannedObjectPtr->N_x; l++)
  {
  	ScannedObjectPtr->Magnetization[j][k][l][0] = MAGOBJECT_INIT_VAL;
  	ScannedObjectPtr->Magnetization[j][k][l][1] = MAGOBJECT_INIT_VAL;
  	ScannedObjectPtr->Magnetization[j][k][l][2] = MAGOBJECT_INIT_VAL;
#ifdef VFET_ELEC_RECON
  	ScannedObjectPtr->ChargeDensity[j][k][l] = ELECOBJECT_INIT_VAL;
#endif
  }

  /*Init = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, PHANTOM_Z_SIZE, PHANTOM_XY_SIZE, PHANTOM_XY_SIZE);
  for (i = 0; i < ScannedObjectPtr->N_time; i++)
  {
	if (read_SharedBinFile_At (, &(ScannedObjectPtr->MagPotentials[i][1][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
 	dwnsmpl_object_bilinear_3D (&(ScannedObjectPtr->MagPotentials[i][1][0][0]), Init, N_z, N_y, N_x, dwnsmpl_factor);
  }
*/
  if (TomoInputsPtr->initICD > 3 || TomoInputsPtr->initICD < 0){
	sentinel(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "ERROR: initICD value not recognized.\n");
  }
  else if (TomoInputsPtr->initICD == 1)
  {
	Read4mBin (MAGNETIZATION_FILENAME, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, 3, sizeof(Real_arr_t), &(ScannedObjectPtr->Magnetization[0][0][0][0]), TomoInputsPtr->debug_file_ptr);
#ifdef VFET_ELEC_RECON
	Read4mBin (ELECCHARGEDENSITY_FILENAME, 1, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, sizeof(Real_arr_t), &(ScannedObjectPtr->ChargeDensity[0][0][0]), TomoInputsPtr->debug_file_ptr);
#endif
/*	size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
	if (read_SharedBinFile_At (MAGNETIZATION_FILENAME, &(ScannedObjectPtr->Magnetization[0][0][0][0]), TomoInputsPtr->node_rank*size*3, size*3, TomoInputsPtr->debug_file_ptr)) flag = -1;
	if (read_SharedBinFile_At (ELECCHARGEDENSITY_FILENAME, &(ScannedObjectPtr->ChargeDensity[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;*/
/*	if (TomoInputsPtr->initMagUpMap == 1)
      	{
		size = TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;
		if (read_SharedBinFile_At (UPDATE_MAP_FILENAME, &(ScannedObjectPtr->UpdateMap[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
      	}*/
  }
  else if (TomoInputsPtr->initICD == 2 || TomoInputsPtr->initICD == 3)
  {
      	if (TomoInputsPtr->initICD == 3)
      	{
        	MagInit = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2, 3);
#ifdef VFET_ELEC_RECON
        	ElecInit = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
#endif
	        check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating object using 3D bilinear interpolation.\n");
			
		/*size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/8;*/
		Read4mBin (MAGNETIZATION_FILENAME, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2, 3, sizeof(Real_arr_t), &(MagInit[0][0][0][0]), TomoInputsPtr->debug_file_ptr);
#ifdef VFET_ELEC_RECON
		Read4mBin (ELECCHARGEDENSITY_FILENAME, 1, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2, sizeof(Real_arr_t), &(ElecInit[0][0][0]), TomoInputsPtr->debug_file_ptr);
#endif
/*		if (read_SharedBinFile_At (MAGNETIZATION_FILENAME, &(MagInit[0][0][0][0]), TomoInputsPtr->node_rank*size*3, size*3, TomoInputsPtr->debug_file_ptr)) flag = -1;
		if (read_SharedBinFile_At (ELECCHARGEDENSITY_FILENAME, &(ElecInit[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;*/
          	upsample_object_bilinear_3D (ScannedObjectPtr->Magnetization, ScannedObjectPtr->ChargeDensity, MagInit, ElecInit, ScannedObjectPtr->N_z/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2, 0);
         		 
		multifree(MagInit, 4);
#ifdef VFET_ELEC_RECON
		multifree(ElecInit, 3);
#endif
        	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Done with interpolating object using 3D bilinear interpolation.\n");
      	}
	else
      	{
        	MagInit = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2, 3);
#ifdef VFET_ELEC_RECON
        	ElecInit = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
#endif
	        check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating object using 2D bilinear interpolation.\n");
/*	  	size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/4;*/
		Read4mBin (MAGNETIZATION_FILENAME, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2, 3, sizeof(Real_arr_t), &(MagInit[0][0][0][0]), TomoInputsPtr->debug_file_ptr);
#ifdef VFET_ELEC_RECON
		Read4mBin (ELECCHARGEDENSITY_FILENAME, 1, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2, sizeof(Real_arr_t), &(ElecInit[0][0][0]), TomoInputsPtr->debug_file_ptr);
#endif
		/*if (read_SharedBinFile_At (MAGNETIZATION_FILENAME, &(MagInit[0][0][0][0]), TomoInputsPtr->node_rank*size*3, size*3, TomoInputsPtr->debug_file_ptr)) flag = -1;
		if (read_SharedBinFile_At (ELECCHARGEDENSITY_FILENAME, &(ElecInit[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;*/
          	upsample_object_bilinear_2D (ScannedObjectPtr->Magnetization, ScannedObjectPtr->ChargeDensity, MagInit, ElecInit, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
        	multifree(MagInit,4);
#ifdef VFET_ELEC_RECON
        	multifree(ElecInit,3);
#endif
        	check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Done with interpolating object using 2D bilinear interpolation.\n");
      	}
/*        if (TomoInputsPtr->initMagUpMap == 1)
        {
          	if (TomoInputsPtr->prevnum_z_blocks == TomoInputsPtr->num_z_blocks)
          	{	
			UpMapInit = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
			size = TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/4;
			if (read_SharedBinFile_At (UPDATE_MAP_FILENAME, &(UpMapInit[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;
          		check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating magnitude update map using 2D bilinear interpolation.\n");
          		upsample_bilinear_2D (ScannedObjectPtr->UpdateMap, UpMapInit, TomoInputsPtr->num_z_blocks, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
          		multifree(UpMapInit,3);
	  	}
		else if (TomoInputsPtr->prevnum_z_blocks == TomoInputsPtr->num_z_blocks/2)
	  	{
			UpMapInit = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, TomoInputsPtr->num_z_blocks/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
			size = TomoInputsPtr->num_z_blocks*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x/8;
			if (read_SharedBinFile_At (UPDATE_MAP_FILENAME, &(UpMapInit[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1; 
          		check_debug(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Interpolating magnitude update map using 3D bilinear interpolation.\n");
			upsample_bilinear_3D (ScannedObjectPtr->UpdateMap, UpMapInit, TomoInputsPtr->num_z_blocks/2, ScannedObjectPtr->N_y/2, ScannedObjectPtr->N_x/2);
          		multifree(UpMapInit,3);
	  	}
	  	else
	  	{
			check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Number of axial blocks is incompatible with previous stage of multi-resolution.\n");
			check_warn(TomoInputsPtr->node_rank==0, TomoInputsPtr->debug_file_ptr, "Initializing the multi-resolution map to zeros.\n");
	  	}	
          }*/
      }
  
    	if (TomoInputsPtr->Write2Tiff == 1)
	{	    	
		dimTiff[3] = 3; dimTiff[0] = ScannedObjectPtr->N_z; dimTiff[1] = ScannedObjectPtr->N_y; dimTiff[2] = ScannedObjectPtr->N_x;
		sprintf (object_file, "%s_n%d", INIT_MAGOBJECT_FILENAME, TomoInputsPtr->node_rank);
    		if (WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 3, 1, 2, &(ScannedObjectPtr->Magnetization[0][0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr))flag = -1;
	    		
#ifdef VFET_ELEC_RECON 
		dimTiff[0] = 1; dimTiff[1] = ScannedObjectPtr->N_z; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;
		sprintf (object_file, "%s_n%d", INIT_ELECOBJECT_FILENAME, TomoInputsPtr->node_rank);
    		if (WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->ChargeDensity[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr))flag = -1;
#endif
	}

	return (flag);
error:
	return (-1);
}


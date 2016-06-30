#include "XT_Constants.h"
#include <stdio.h>
#include "XT_Structures.h"
#include "XT_Prior.h"
#include "XT_AMatrix.h"
#include <math.h>
#include "allocate.h"

/*finds the maximum in a array 'array_in' with number of elements being 'num'*/
int32_t find_max(int32_t* array_in, int32_t num)
{
  int32_t i, maxnum;
  maxnum = array_in[0];
  for (i=1; i<num; i++)
  if (array_in[i] > maxnum)
  maxnum = array_in[i];
  
  return(maxnum);
}

void compute_voxel_update_Atten (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** ErrorSino_Unflip_x, Real_arr_t*** ErrorSino_Flip_x, Real_arr_t*** ErrorSino_Unflip_y, Real_arr_t*** ErrorSino_Flip_y, AMatrixCol* AMatrixPtr_X, AMatrixCol* AMatrixPtr_Y, Real_t MagPrior[3], Real_t ElecPrior, int32_t slice, int32_t j_new, int32_t k_new)
{
  	int32_t p, q, r, z_overlap_num;
	Real_t VMag[3], THETA1Mag[3], THETA2Mag[3][3], ProjectionEntry, ProjEntryComp, THETA1Elec, THETA2Elec;
  	int32_t i_r, i_t;

        VMag[0] = ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0]; /*Store the present value of the voxel*/
        VMag[1] = ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1]; /*Store the present value of the voxel*/
        VMag[2] = ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2]; /*Store the present value of the voxel*/

	z_overlap_num = SinogramPtr->z_overlap_num;

	THETA1Mag[0] = 0.0;
	THETA1Mag[1] = 0.0;
	THETA1Mag[2] = 0.0;

	THETA2Mag[0][0] = 0.0; THETA2Mag[0][1] = 0.0; THETA2Mag[0][2] = 0.0;
	THETA2Mag[1][0] = 0.0; THETA2Mag[1][1] = 0.0; THETA2Mag[1][2] = 0.0;
	THETA2Mag[2][0] = 0.0; THETA2Mag[2][1] = 0.0; THETA2Mag[2][2] = 0.0;

#ifdef VFET_ELEC_RECON
	Real_t VElec;
        VElec = ScannedObjectPtr->ElecPotentials[slice][j_new][k_new]; /*Store the present value of the voxel*/
	THETA1Elec = 0.0;
	THETA2Elec = 0.0;
#endif	

	for (p = 0; p < SinogramPtr->N_p; p++){
		for (q = 0; q < AMatrixPtr_X[p].count; q++)
		{
      		    	i_r = (AMatrixPtr_X[p].index[q]);
       		    	ProjectionEntry = (AMatrixPtr_X[p].values[q]);
/*      	 	ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];*/
				i_t = k_new*z_overlap_num + r;
				THETA2Mag[0][1] += -ProjectionEntry*ProjectionEntry*SinogramPtr->sine[p]*SinogramPtr->cosine[p]*TomoInputsPtr->Weight*TomoInputsPtr->MagPhaseMultiple*TomoInputsPtr->MagPhaseMultiple;
					
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        	   	THETA2Mag[0][0] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[0] += -((ErrorSino_Unflip_x[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
				THETA2Mag[1][1] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[1] += -((ErrorSino_Unflip_x[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
#ifdef VFET_ELEC_RECON				
				THETA2Mag[0][1] += -ProjectionEntry*ProjectionEntry*SinogramPtr->sine[p]*SinogramPtr->cosine[p]*TomoInputsPtr->Weight*TomoInputsPtr->MagPhaseMultiple*TomoInputsPtr->MagPhaseMultiple;
					
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        	   	THETA2Mag[0][0] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[0] += -((-ErrorSino_Flip_x[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
				THETA2Mag[1][1] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[1] += -((-ErrorSino_Flip_x[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = ProjectionEntry*TomoInputsPtr->ElecPhaseMultiple;
	        	   	THETA2Elec += 2*ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Elec += -((ErrorSino_Unflip_x[p][i_r][i_t]+ErrorSino_Flip_x[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
#endif
            		}
		}
        }
	THETA2Mag[1][0] = THETA2Mag[0][1];
	
	for (p = 0; p < SinogramPtr->N_p; p++){
		for (q = 0; q < AMatrixPtr_Y[p].count; q++)
		{
      		    	i_r = (AMatrixPtr_Y[p].index[q]);
       		    	ProjectionEntry = (AMatrixPtr_Y[p].values[q]);
/*      	 	ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];*/
				i_t = j_new*z_overlap_num + r;
				THETA2Mag[0][2] += -ProjectionEntry*ProjectionEntry*SinogramPtr->sine[p]*SinogramPtr->cosine[p]*TomoInputsPtr->Weight*TomoInputsPtr->MagPhaseMultiple*TomoInputsPtr->MagPhaseMultiple;
					
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        	   	THETA2Mag[0][0] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[0] += -((ErrorSino_Unflip_y[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
				THETA2Mag[2][2] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[2] += -((ErrorSino_Unflip_y[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
#ifdef VFET_ELEC_RECON				
				THETA2Mag[0][2] += -ProjectionEntry*ProjectionEntry*SinogramPtr->sine[p]*SinogramPtr->cosine[p]*TomoInputsPtr->Weight*TomoInputsPtr->MagPhaseMultiple*TomoInputsPtr->MagPhaseMultiple;
					
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        	   	THETA2Mag[0][0] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[0] += -((-ErrorSino_Flip_y[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
				THETA2Mag[2][2] += ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[2] += -((-ErrorSino_Flip_y[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = ProjectionEntry*TomoInputsPtr->ElecPhaseMultiple;
	        	   	THETA2Elec += 2*ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Elec += -((ErrorSino_Unflip_y[p][i_r][i_t]+ErrorSino_Flip_y[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
#endif
            		}
		}
        }
	THETA2Mag[2][0] = THETA2Mag[0][2];

            /*Solve the 1-D optimization problem
            TODO : What if theta1 = 0 ? Then this will give error*/

	MagFunctionalSubstitutionConstPrior(&(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0]), THETA1Mag, THETA2Mag, ScannedObjectPtr, TomoInputsPtr, MagPrior);

#ifdef VFET_ELEC_RECON
	ElecFunctionalSubstitutionConstPrior(&(ScannedObjectPtr->ElecPotentials[slice][j_new][k_new]), THETA1Elec, THETA2Elec, ScannedObjectPtr, TomoInputsPtr, ElecPrior);
#endif
	
	for (p = 0; p < SinogramPtr->N_p; p++){
		for (q = 0; q < AMatrixPtr_X[p].count; q++)
        	{
               	    	i_r = (AMatrixPtr_X[p].index[q]);
        	    	ProjectionEntry = (AMatrixPtr_X[p].values[q]);
        	    	/*ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];
	        		ErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*VoxelLineResponse[slice].values[r]*(ScannedObjectPtr->Object[i_new][slice+1][j_new][k_new] - V));*/
				i_t = k_new*z_overlap_num + r;
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_x[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]));

				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_x[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1] - VMag[1]));
				
#ifdef VFET_ELEC_RECON				
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Flip_x[p][i_r][i_t] -= (-ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]));
				
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Flip_x[p][i_r][i_t] -= (-ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1] - VMag[1]));
				
				ProjEntryComp = ProjectionEntry*TomoInputsPtr->ElecPhaseMultiple;
	        		ErrorSino_Unflip_x[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->ElecPotentials[slice][j_new][k_new] - VElec));
	        		ErrorSino_Flip_x[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->ElecPotentials[slice][j_new][k_new] - VElec));
#endif
	   		}
		}
	}
	
	for (p = 0; p < SinogramPtr->N_p; p++){
		for (q = 0; q < AMatrixPtr_Y[p].count; q++)
        	{
               	    	i_r = (AMatrixPtr_Y[p].index[q]);
        	    	ProjectionEntry = (AMatrixPtr_Y[p].values[q]);
        	    	/*ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];
	        		ErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*VoxelLineResponse[slice].values[r]*(ScannedObjectPtr->Object[i_new][slice+1][j_new][k_new] - V));*/
				i_t = j_new*z_overlap_num + r;
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_y[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]));
			
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_y[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2] - VMag[2]));
	
#ifdef VFET_ELEC_RECON				
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Flip_y[p][i_r][i_t] -= (-ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]));
				
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p]*TomoInputsPtr->MagPhaseMultiple;
	        		ErrorSino_Flip_y[p][i_r][i_t] -= (-ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2] - VMag[2]));
				
				ProjEntryComp = ProjectionEntry*TomoInputsPtr->ElecPhaseMultiple;
	        		ErrorSino_Unflip_y[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->ElecPotentials[slice][j_new][k_new] - VElec));
	        		ErrorSino_Flip_y[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->ElecPotentials[slice][j_new][k_new] - VElec));
#endif
	   		}
		}
	}
	
	ScannedObjectPtr->ErrorPotMag[slice][j_new][k_new][0] += (ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]); 
	ScannedObjectPtr->ErrorPotMag[slice][j_new][k_new][1] += (ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1] - VMag[1]); 
	ScannedObjectPtr->ErrorPotMag[slice][j_new][k_new][2] += (ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2] - VMag[2]); 
#ifdef VFET_ELEC_RECON				
	ScannedObjectPtr->ErrorPotElec[slice][j_new][k_new] += (ScannedObjectPtr->ElecPotentials[slice][j_new][k_new] - VElec);
#endif
}

Real_t updateVoxels_Atten (int32_t slice_begin, int32_t slice_end, int32_t xy_begin, int32_t xy_end, int32_t* x_rand_select, int32_t* y_rand_select, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** ErrorSino_Unflip_x, Real_arr_t*** ErrorSino_Flip_x, Real_arr_t*** ErrorSino_Unflip_y, Real_arr_t*** ErrorSino_Flip_y, Real_arr_t** DetectorResponse_XY, /*AMatrixCol* VoxelLineResponse,*/ int32_t Iter, long int *zero_count, Real_t *MagUpdate, Real_t *ElecUpdate, Real_t *MagSum, Real_t *ElecSum, uint8_t** Mask)
{
  int32_t p,slice,j_new,k_new,index_xy;
  Real_t VMag[3], MagPrior[3];
  int32_t z_min, z_max;
  Real_t total_vox_mag = 0.0;
  Real_t VElec, ElecPrior;

  z_min = 0;
  z_max = ScannedObjectPtr->N_z - 1;

   /*printf ("maxview = %d, size of AMatrixCol = %d\n",maxview,sizeof(AMatrixCol));*/
  AMatrixCol* AMatrixPtr_X = (AMatrixCol*)get_spc(SinogramPtr->N_p, sizeof(AMatrixCol));
  AMatrixCol* AMatrixPtr_Y = (AMatrixCol*)get_spc(SinogramPtr->N_p, sizeof(AMatrixCol));
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
  
  for (p = 0; p < SinogramPtr->N_p; p++)
  {
  	AMatrixPtr_X[p].values = (Real_t*)get_spc(AvgNumXElements,sizeof(Real_t));
  	AMatrixPtr_X[p].index  = (int32_t*)get_spc(AvgNumXElements,sizeof(int32_t));
  	AMatrixPtr_Y[p].values = (Real_t*)get_spc(AvgNumXElements,sizeof(Real_t));
  	AMatrixPtr_Y[p].index  = (int32_t*)get_spc(AvgNumXElements,sizeof(int32_t));
  }  

      for (slice = slice_begin; slice <= slice_end; slice++) {
      for (index_xy = xy_begin; index_xy <= xy_end; index_xy++) 
      {
    /*    printf ("Entering index\n");*/ 
	k_new = x_rand_select[index_xy];
        j_new = y_rand_select[index_xy];
    	/*MagUpdateMap[j_new][k_new] = 0; */ 
           /*printf ("Entering mask\n"); */
	  /*for (p = 0; p < SinogramPtr->N_p; p++)
    	  {
		calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, DetectorResponse_XY, &(AMatrixPtr_X[p]), j_new, k_new, p);
    	  }*/
	  	for (p = 0; p < SinogramPtr->N_p; p++)
    	  	{
			/*calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, DetectorResponse_XY, &(AMatrixPtr_Y[p]), j_new, slice, p);*/
			calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, DetectorResponse_XY, &(AMatrixPtr_X[p]), slice, j_new, p);
			calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, DetectorResponse_XY, &(AMatrixPtr_Y[p]), slice, k_new, p);
		}
        /*  	printf ("Entering slice\n");*/ 
            /*For a given (i,j,k) store its 26 point neighborhood*/           
	    if (Mask[j_new][k_new] == 1)
	    {   
		MagPrior[0] = -(ScannedObjectPtr->ErrorPotMag[slice][j_new][k_new][0] - ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0]);
		MagPrior[1] = -(ScannedObjectPtr->ErrorPotMag[slice][j_new][k_new][1] - ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1]);
		MagPrior[2] = -(ScannedObjectPtr->ErrorPotMag[slice][j_new][k_new][2] - ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2]); 
        
		VMag[0] = ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0]; /*Store the present value of the voxel*/
        	VMag[1] = ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1]; /*Store the present value of the voxel*/
        	VMag[2] = ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2]; /*Store the present value of the voxel*/

#ifdef VFET_ELEC_RECON		
		ElecPrior = -(ScannedObjectPtr->ErrorPotElec[slice][j_new][k_new] - ScannedObjectPtr->ElecPotentials[slice][j_new][k_new]); 
        	VElec = ScannedObjectPtr->ElecPotentials[slice][j_new][k_new]; /*Store the present value of the voxel*/
#endif

		compute_voxel_update_Atten (SinogramPtr, ScannedObjectPtr, TomoInputsPtr, ErrorSino_Unflip_x, ErrorSino_Flip_x, ErrorSino_Unflip_y, ErrorSino_Flip_y, AMatrixPtr_X, AMatrixPtr_Y, MagPrior, ElecPrior, slice, j_new, k_new);

		(*MagUpdate) += sqrt(pow(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0],2) + pow(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1] - VMag[1],2) + pow(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2] - VMag[2],2)); 
		(*MagSum) += sqrt(pow(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][0],2) + pow(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][1],2) + pow(ScannedObjectPtr->MagPotentials[slice][j_new][k_new][2],2)); 

#ifdef VFET_ELEC_RECON		
		(*ElecUpdate) += fabs(ScannedObjectPtr->ElecPotentials[slice][j_new][k_new] - VElec); 
		(*ElecSum) += fabs(ScannedObjectPtr->ElecPotentials[slice][j_new][k_new]);
#endif
       	     }
       }
       }

    
     for (p=0; p<SinogramPtr->N_p; p++)
     {
     	free(AMatrixPtr_X[p].values);
     	free(AMatrixPtr_X[p].index);
     	free(AMatrixPtr_Y[p].values);
     	free(AMatrixPtr_Y[p].index);
     }
     free(AMatrixPtr_X);
     free(AMatrixPtr_Y);
      return (total_vox_mag);
}


Real_t updateVoxels (int32_t slice_begin, int32_t slice_end, int32_t xy_begin, int32_t xy_end, int32_t* x_rand_select, int32_t* y_rand_select, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** ErrorSino_Unflip_x, Real_arr_t*** ErrorSino_Flip_x, Real_arr_t*** ErrorSino_Unflip_y, Real_arr_t*** ErrorSino_Flip_y, Real_arr_t** DetectorResponse_XY, int32_t Iter, long int *zero_count, Real_t *MagUpdate, Real_t *ElecUpdate, Real_t *MagSum, Real_t *ElecSum, uint8_t** Mask)
{
	Real_t total_vox_mag;
	total_vox_mag = updateVoxels_Atten(slice_begin, slice_end, xy_begin, xy_end, x_rand_select, y_rand_select, SinogramPtr, ScannedObjectPtr, TomoInputsPtr, ErrorSino_Unflip_x, ErrorSino_Flip_x, ErrorSino_Unflip_y, ErrorSino_Flip_y, DetectorResponse_XY, Iter, zero_count, MagUpdate, ElecUpdate, MagSum, ElecSum, Mask);
	return (total_vox_mag);
}

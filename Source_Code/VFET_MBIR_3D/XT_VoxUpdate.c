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

void compute_voxel_update_Atten (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** ErrorSino_Unflip_z, Real_arr_t*** ErrorSino_Flip_z, AMatrixCol* AMatrixPtr, /*AMatrixCol* VoxelLineResponse,*/ Real_t Mag3D_Nhood[3][3][3][2], Real_t Elec3D_Nhood[3][3][3], bool BDFlag_3D[3][3][3], int32_t slice, int32_t j_new, int32_t k_new)
{
  	int32_t p, q, r, z_overlap_num;
	Real_t VMag[2], VElec, THETA1Mag[2], THETA1Elec, THETA2Mag[2][2], THETA2Elec, ProjectionEntry, ProjEntryComp;
  	int32_t i_r, i_t;

        VMag[0] = ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][0]; /*Store the present value of the voxel*/
        VMag[1] = ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][1]; /*Store the present value of the voxel*/
        VElec = ScannedObjectPtr->ElecPotentials[slice+1][j_new][k_new]; /*Store the present value of the voxel*/

	z_overlap_num = SinogramPtr->z_overlap_num;

	THETA1Mag[0] = 0.0;
	THETA1Mag[1] = 0.0;
	THETA1Elec = 0.0;

	THETA2Mag[0][0] = 0.0; THETA2Mag[0][1] = 0.0;
	THETA2Mag[1][0] = 0.0; THETA2Mag[1][1] = 0.0;
	THETA2Elec = 0.0;

	for (p = 0; p < SinogramPtr->N_p; p++){
		for (q = 0; q < AMatrixPtr[p].count; q++)
		{
      		    	i_r = (AMatrixPtr[p].index[q]);
       		    	ProjectionEntry = (AMatrixPtr[p].values[q]);
/*      	 	ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];*/
				i_t = slice*z_overlap_num + r;
				THETA2Mag[0][1] += -2*ProjectionEntry*ProjectionEntry*SinogramPtr->sine[p]*SinogramPtr->cosine[p]*TomoInputsPtr->Weight;
					
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p];
	        	   	THETA2Mag[0][0] += 2*ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[0] += -((ErrorSino_Unflip_z[p][i_r][i_t]-ErrorSino_Flip_z[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p];
				THETA2Mag[1][1] += 2*ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Mag[1] += -((ErrorSino_Unflip_z[p][i_r][i_t]-ErrorSino_Flip_z[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
				
				ProjEntryComp = ProjectionEntry;
	        	   	THETA2Elec += 2*ProjEntryComp*ProjEntryComp*TomoInputsPtr->Weight;
               			THETA1Elec += -((ErrorSino_Unflip_z[p][i_r][i_t]+ErrorSino_Flip_z[p][i_r][i_t])*ProjEntryComp*TomoInputsPtr->Weight);
            		}
		}
        }
	THETA2Mag[1][0] = THETA2Mag[0][1];

            /*Solve the 1-D optimization problem
            TODO : What if theta1 = 0 ? Then this will give error*/

/*	ScannedObjectPtr->MagPotentials[i_new][slice+1][j_new][k_new] = Mag_FunctionalSubstitution(ScannedObjectPtr->MagPotentials[i_new][slice+1][j_new][k_new], THETA1Mag, THETA2, ScannedObjectPtr, TomoInputsPtr, Mag3D_Nhood, MagTime_Nhood, BDFlag_3D, BDFlag_Time);
        ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new] = Phase_FunctionalSubstitution(ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new], THETA1Phase, THETA2, ScannedObjectPtr, TomoInputsPtr, Phase3D_Nhood, PhaseTime_Nhood, BDFlag_3D, BDFlag_Time);*/
	FunctionalSubstitution(&(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][0]), &(ScannedObjectPtr->ElecPotentials[slice+1][j_new][k_new]), THETA1Mag, THETA2Mag, THETA1Elec, THETA2Elec, ScannedObjectPtr, TomoInputsPtr, Mag3D_Nhood, Elec3D_Nhood, BDFlag_3D);
	
	for (p = 0; p < SinogramPtr->N_p; p++){
		for (q = 0; q < AMatrixPtr[p].count; q++)
        	{
               	    	i_r = (AMatrixPtr[p].index[q]);
        	    	ProjectionEntry = (AMatrixPtr[p].values[q]);
        	    	/*ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];
	        		ErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*VoxelLineResponse[slice].values[r]*(ScannedObjectPtr->Object[i_new][slice+1][j_new][k_new] - V));*/
				i_t = slice*z_overlap_num + r;
				ProjEntryComp = ProjectionEntry*SinogramPtr->cosine[p];
	        		ErrorSino_Unflip_z[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][0] - VMag[0]));
	        		ErrorSino_Flip_z[p][i_r][i_t] -= (-ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][0] - VMag[0]));
				ProjEntryComp = -ProjectionEntry*SinogramPtr->sine[p];
	        		ErrorSino_Unflip_z[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][1] - VMag[1]));
	        		ErrorSino_Flip_z[p][i_r][i_t] -= (-ProjEntryComp*(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][1] - VMag[1]));
				ProjEntryComp = ProjectionEntry;
	        		ErrorSino_Unflip_z[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->ElecPotentials[slice+1][j_new][k_new] - VElec));
	        		ErrorSino_Flip_z[p][i_r][i_t] -= (ProjEntryComp*(ScannedObjectPtr->ElecPotentials[slice+1][j_new][k_new] - VElec));
	   		}
		}
	}
}

Real_t updateVoxels_Atten (int32_t slice_begin, int32_t slice_end, int32_t xy_begin, int32_t xy_end, int32_t* x_rand_select, int32_t* y_rand_select, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** ErrorSino_Unflip_z, Real_arr_t*** ErrorSino_Flip_z, Real_arr_t** DetectorResponse_XY, /*AMatrixCol* VoxelLineResponse,*/ int32_t Iter, long int *zero_count, Real_arr_t** MagUpdateMap, uint8_t** Mask)
{
  int32_t p,q,r,slice,j_new,k_new,idxr,idxq,idxp,index_xy;
  Real_t VMag[3], VElec;
  bool ZSFlag;
  int32_t z_min, z_max;
  Real_t total_vox_mag = 0.0;

  z_min = 0;
  z_max = ScannedObjectPtr->N_z + 1;
  if (TomoInputsPtr->node_rank == 0)
	z_min = 1;
  if (TomoInputsPtr->node_rank == TomoInputsPtr->node_num - 1)
	z_max = ScannedObjectPtr->N_z;

    Real_t Mag3D_Nhood[3][3][3][2]; 
    Real_t Elec3D_Nhood[3][3][3]; 
    bool BDFlag_3D[3][3][3];

   /*printf ("maxview = %d, size of AMatrixCol = %d\n",maxview,sizeof(AMatrixCol));*/
  AMatrixCol* AMatrixPtr = (AMatrixCol*)get_spc(SinogramPtr->N_p, sizeof(AMatrixCol));
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
  
  for (p = 0; p < SinogramPtr->N_p; p++)
  {
  	AMatrixPtr[p].values = (Real_t*)get_spc(AvgNumXElements,sizeof(Real_t));
  	AMatrixPtr[p].index  = (int32_t*)get_spc(AvgNumXElements,sizeof(int32_t));
  }  

      for (index_xy = xy_begin; index_xy <= xy_end; index_xy++) 
      {
    /*    printf ("Entering index\n");*/ 
	k_new = x_rand_select[index_xy];
        j_new = y_rand_select[index_xy];
    	MagUpdateMap[j_new][k_new] = 0;  
           /*printf ("Entering mask\n"); */
	  for (p = 0; p < SinogramPtr->N_p; p++)
    	  {
		calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, DetectorResponse_XY, &(AMatrixPtr[p]), j_new, k_new, p);
    	  }
          for (slice = slice_begin; slice <= slice_end; slice++) {
        /*  	printf ("Entering slice\n");*/ 
            /*For a given (i,j,k) store its 26 point neighborhood*/           
	    if (Mask[j_new][k_new] == 1)
	    {   
	 	for (p = 0; p < 3; p++)
	 	{
			idxp = slice + p;
			if (idxp >= z_min && idxp <= z_max)
			{
	 			for (q = 0; q < 3; q++)
         			{
	 				idxq = j_new + q - 1;
                			if(idxq >= 0 && idxq < ScannedObjectPtr->N_y)
         				{
						for (r = 0; r < 3; r++)
						{
		    					idxr = k_new + r - 1;
                    					if(idxr >= 0 && idxr < ScannedObjectPtr->N_x){
	                					Mag3D_Nhood[p][q][r][0] = ScannedObjectPtr->MagPotentials[idxp][idxq][idxr][0];
	                					Mag3D_Nhood[p][q][r][1] = ScannedObjectPtr->MagPotentials[idxp][idxq][idxr][1];
	                					Elec3D_Nhood[p][q][r] = ScannedObjectPtr->ElecPotentials[idxp][idxq][idxr];
        	        					BDFlag_3D[p][q][r] = true;
                    					}
							else
							{
	                					Mag3D_Nhood[p][q][r][0] = 0.0;
	                					Mag3D_Nhood[p][q][r][1] = 0.0;
	                					Elec3D_Nhood[p][q][r] = 0.0;
                    						BDFlag_3D[p][q][r] = false;
							}
						}
					}
		 			else
					{
         					for (r = 0; r < 3; r++){
	                				Mag3D_Nhood[p][q][r][0] = 0.0;
	                				Mag3D_Nhood[p][q][r][1] = 0.0;
	                				Elec3D_Nhood[p][q][r] = 0.0;
                    					BDFlag_3D[p][q][r] = false;
						}
					}
                		}
			}
			else
        		{ 
				for (q = 0; q < 3; q++){
				for (r = 0; r < 3; r++){
	              			Mag3D_Nhood[p][q][r][0] = 0.0;
	              			Mag3D_Nhood[p][q][r][1] = 0.0;
	              			Elec3D_Nhood[p][q][r] = 0.0;
                   			BDFlag_3D[p][q][r] = false;
				}
				}
               		}
		}

        Mag3D_Nhood[1][1][1][0] = 0.0;
        Mag3D_Nhood[1][1][1][1] = 0.0;
        Elec3D_Nhood[1][1][1] = 0.0;
        VMag[0] = ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][0]; /*Store the present value of the voxel*/
        VMag[1] = ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][1]; /*Store the present value of the voxel*/
        VElec = ScannedObjectPtr->ElecPotentials[slice+1][j_new][k_new]; /*Store the present value of the voxel*/

#ifdef ZERO_SKIPPING
			  /*Zero Skipping Algorithm*/
			 ZSFlag = true;
			 if(VMag[0] == 0.0 && VMag[1] == 0.0 && VElec == 0.0 && Iter > 1) /*Iteration starts from 1. Iteration 0 corresponds to initial cost before ICD*/
			  {
					for(p = 0; p < 3; p++)
						for(q = 0; q < 3; q++)
					  		for(r = 0; r < 3; r++)
							  	if(Mag3D_Nhood[p][q][r][0] > 0.0 || Mag3D_Nhood[p][q][r][1] > 0.0 || Elec3D_Nhood[p][q][r] > 0.0)
							  	{
									  ZSFlag = false;
								 	  break;
							  	}
			  }
			  else
			  {
				  ZSFlag = false;
			  }
#else
			  ZSFlag = false; /*do ICD on all voxels*/
#endif /*#ifdef ZERO_SKIPPING*/
	if(ZSFlag == false)
	{
		compute_voxel_update_Atten (SinogramPtr, ScannedObjectPtr, TomoInputsPtr, ErrorSino_Unflip_z, ErrorSino_Flip_z, AMatrixPtr, /*VoxelLineResponse,*/ Mag3D_Nhood, Elec3D_Nhood, BDFlag_3D, slice, j_new, k_new);
	    	MagUpdateMap[j_new][k_new] += sqrt(pow(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][0] - VMag[0],2) + pow(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][1] - VMag[1],2) + pow(ScannedObjectPtr->ElecPotentials[slice+1][j_new][k_new] - VElec,2));
	    	total_vox_mag += sqrt(pow(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][0],2) + pow(ScannedObjectPtr->MagPotentials[slice+1][j_new][k_new][1],2) + pow(ScannedObjectPtr->ElecPotentials[slice+1][j_new][k_new],2));
 	}
		else
		    (*zero_count)++;
       }
       }
       }

    
     for (p=0; p<SinogramPtr->N_p; p++)
     {
     	free(AMatrixPtr[p].values);
     	free(AMatrixPtr[p].index);
     }
     free(AMatrixPtr);
      return (total_vox_mag);
}


Real_t updateVoxels (int32_t slice_begin, int32_t slice_end, int32_t xy_begin, int32_t xy_end, int32_t* x_rand_select, int32_t* y_rand_select, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** ErrorSino_Unflip_z, Real_arr_t*** ErrorSino_Flip_z, Real_arr_t** DetectorResponse_XY, int32_t Iter, long int *zero_count, Real_arr_t** MagUpdateMap, uint8_t** Mask)
{
	Real_t total_vox_mag;
	total_vox_mag = updateVoxels_Atten(slice_begin, slice_end, xy_begin, xy_end, x_rand_select, y_rand_select, SinogramPtr, ScannedObjectPtr, TomoInputsPtr, ErrorSino_Unflip_z, ErrorSino_Flip_z, DetectorResponse_XY, Iter, zero_count, MagUpdateMap, Mask);
	return (total_vox_mag);
}

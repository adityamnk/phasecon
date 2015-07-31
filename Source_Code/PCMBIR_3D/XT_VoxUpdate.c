

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

void compute_voxel_update_Atten (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** MagErrorSino, Real_arr_t*** PhaseErrorSino, AMatrixCol* AMatrixPtr, /*AMatrixCol* VoxelLineResponse,*/ Real_t Mag3D_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Phase3D_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t MagTime_Nhood[], Real_t PhaseTime_Nhood[], bool BDFlag_3D[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], bool BDFlag_Time[], int32_t i_new, int32_t slice, int32_t j_new, int32_t k_new)
{
  	int32_t p, q, r, sino_view, z_overlap_num;
	Real_t VMag,VPhase,THETA1Mag,THETA1Phase,THETA2;
	Real_t UpdatedVoxelValue, ProjectionEntry;
  	int32_t i_r, i_t;
        VMag = ScannedObjectPtr->MagObject[i_new][slice+1][j_new][k_new]; /*Store the present value of the voxel*/
        VPhase = ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new]; /*Store the present value of the voxel*/
	z_overlap_num = SinogramPtr->z_overlap_num;

	THETA1Mag = 0.0;
	THETA1Phase = 0.0;
	THETA2 = 0.0;
	for (p = 0; p < ScannedObjectPtr->ProjNum[i_new]; p++){
		sino_view = ScannedObjectPtr->ProjIdxPtr[i_new][p];
		for (q = 0; q < AMatrixPtr[p].count; q++)
		{
      		    	i_r = (AMatrixPtr[p].index[q]);
       		    	ProjectionEntry = (AMatrixPtr[p].values[q]*SinogramPtr->delta_t);
/*      	 	ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];*/
				i_t = slice*z_overlap_num + r;
	        	   	THETA2 += ProjectionEntry*ProjectionEntry*TomoInputsPtr->ADMM_mu;
               			THETA1Mag += -(MagErrorSino[sino_view][i_r][i_t]*ProjectionEntry*TomoInputsPtr->ADMM_mu);
               			THETA1Phase += -(PhaseErrorSino[sino_view][i_r][i_t]*ProjectionEntry*TomoInputsPtr->ADMM_mu);
            		}
		}
        }

            /*Solve the 1-D optimization problem
            TODO : What if theta1 = 0 ? Then this will give error*/

        UpdatedVoxelValue = Mag_FunctionalSubstitution(VMag, THETA1Mag, THETA2, ScannedObjectPtr, TomoInputsPtr, Mag3D_Nhood, MagTime_Nhood, BDFlag_3D, BDFlag_Time);
        ScannedObjectPtr->MagObject[i_new][slice+1][j_new][k_new] = UpdatedVoxelValue;
        UpdatedVoxelValue = Phase_FunctionalSubstitution(VPhase, THETA1Phase, THETA2, ScannedObjectPtr, TomoInputsPtr, Phase3D_Nhood, PhaseTime_Nhood, BDFlag_3D, BDFlag_Time);
        ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new] = UpdatedVoxelValue;
        /*ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new] = 0;*/
	
	for (p = 0; p < ScannedObjectPtr->ProjNum[i_new]; p++){
		sino_view = ScannedObjectPtr->ProjIdxPtr[i_new][p];
		for (q = 0; q < AMatrixPtr[p].count; q++)
        	{
               	    	i_r = (AMatrixPtr[p].index[q]);
        	    	ProjectionEntry = (AMatrixPtr[p].values[q]*SinogramPtr->delta_t);
        	    	/*ProjectionEntry = (AMatrixPtr[p].values[q]);
			for (r = 0; r < VoxelLineResponse[slice].count; r++)*/
			for (r = 0; r < z_overlap_num; r++)
			{ 
				/*i_t = VoxelLineResponse[slice].index[r];
	        		ErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*VoxelLineResponse[slice].values[r]*(ScannedObjectPtr->Object[i_new][slice+1][j_new][k_new] - V));*/
				i_t = slice*z_overlap_num + r;
	        		MagErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*(ScannedObjectPtr->MagObject[i_new][slice+1][j_new][k_new] - VMag));
	        		PhaseErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*(ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new] - VPhase));
	   		}
		}
	}
}

Real_t updateVoxels_Atten (int32_t time_begin, int32_t time_end, int32_t slice_begin, int32_t slice_end, int32_t xy_begin, int32_t xy_end, int32_t* x_rand_select, int32_t* y_rand_select, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** MagErrorSino, Real_arr_t*** PhaseErrorSino, Real_arr_t** DetectorResponse_XY, /*AMatrixCol* VoxelLineResponse,*/ int32_t Iter, long int *zero_count, Real_arr_t** MagUpdateMap, uint8_t** Mask)
{
  int32_t p,q,r,slice,i_new,j_new,k_new,idxr,idxq,idxp,index_xy;
  Real_t VMag, VPhase;
  bool ZSFlag;
  int32_t sino_view;
  int32_t z_min, z_max;
  Real_t total_vox_mag = 0.0;

  z_min = 0;
  z_max = ScannedObjectPtr->N_z + 1;
  if (TomoInputsPtr->node_rank == 0)
	z_min = 1;
  if (TomoInputsPtr->node_rank == TomoInputsPtr->node_num - 1)
	z_max = ScannedObjectPtr->N_z;

    Real_t Mag3D_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM]; 
    Real_t MagTime_Nhood[NHOOD_TIME_MAXDIM-1]; 
    Real_t Phase3D_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM]; 
    Real_t PhaseTime_Nhood[NHOOD_TIME_MAXDIM-1]; 
    bool BDFlag_3D[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM];
    bool BDFlag_Time[NHOOD_TIME_MAXDIM-1];

  int32_t maxview = find_max(ScannedObjectPtr->ProjNum, ScannedObjectPtr->N_time);
   /*printf ("maxview = %d, size of AMatrixCol = %d\n",maxview,sizeof(AMatrixCol));*/
  AMatrixCol* AMatrixPtr = (AMatrixCol*)get_spc(maxview, sizeof(AMatrixCol));
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ScannedObjectPtr->delta_xy/SinogramPtr->delta_r);
  
  for (p = 0; p < maxview; p++)
  {
  	AMatrixPtr[p].values = (Real_t*)get_spc(AvgNumXElements,sizeof(Real_t));
  	AMatrixPtr[p].index  = (int32_t*)get_spc(AvgNumXElements,sizeof(int32_t));
  }  

   for (i_new = time_begin; i_new <= time_end; i_new++) 
   {
      for (index_xy = xy_begin; index_xy <= xy_end; index_xy++) 
      {
    /*    printf ("Entering index\n");*/ 
	k_new = x_rand_select[index_xy];
        j_new = y_rand_select[index_xy];
    	MagUpdateMap[j_new][k_new] = 0;  
           /*printf ("Entering mask\n"); */
	  for (p = 0; p < ScannedObjectPtr->ProjNum[i_new]; p++)
    	  {
		sino_view = ScannedObjectPtr->ProjIdxPtr[i_new][p];
		calcAMatrixColumnforAngle(SinogramPtr, ScannedObjectPtr, DetectorResponse_XY, &(AMatrixPtr[p]), j_new, k_new, sino_view);
    	  }
          for (slice = slice_begin; slice <= slice_end; slice++) {
        /*  	printf ("Entering slice\n");*/ 
            /*For a given (i,j,k) store its 26 point neighborhood*/           
	    if (Mask[j_new][k_new] == 1)
	    {   
	 	if (i_new - 1 >= 0){
			MagTime_Nhood[0] = ScannedObjectPtr->MagObject[i_new-1][slice+1][j_new][k_new];
			PhaseTime_Nhood[0] = ScannedObjectPtr->PhaseObject[i_new-1][slice+1][j_new][k_new];
			BDFlag_Time[0] = true;
		}
		else 
		{
			MagTime_Nhood[0] = 0.0;
			PhaseTime_Nhood[0] = 0.0;
			BDFlag_Time[0] = false;
		}

	    	if (i_new + 1 < ScannedObjectPtr->N_time){
			MagTime_Nhood[1] = ScannedObjectPtr->MagObject[i_new+1][slice+1][j_new][k_new];
			PhaseTime_Nhood[1] = ScannedObjectPtr->PhaseObject[i_new+1][slice+1][j_new][k_new];
			BDFlag_Time[1] = true;
		}
		else
		{
			MagTime_Nhood[1] = 0.0;
			PhaseTime_Nhood[1] = 0.0;
			BDFlag_Time[1] = false;
		}
	
	
	 for (p = 0; p < NHOOD_Z_MAXDIM; p++)
	 {
		idxp = slice + p;
		if (idxp >= z_min && idxp <= z_max)
		{
	 		for (q = 0; q < NHOOD_Y_MAXDIM; q++)
         		{
	 			idxq = j_new + q - 1;
                		if(idxq >= 0 && idxq < ScannedObjectPtr->N_y)
         			{
					for (r = 0; r < NHOOD_X_MAXDIM; r++)
					{
		    				idxr = k_new + r - 1;
                    				if(idxr >= 0 && idxr < ScannedObjectPtr->N_x){
	                				Mag3D_Nhood[p][q][r] = ScannedObjectPtr->MagObject[i_new][idxp][idxq][idxr];
	                				Phase3D_Nhood[p][q][r] = ScannedObjectPtr->PhaseObject[i_new][idxp][idxq][idxr];
        	        				BDFlag_3D[p][q][r] = true;
                    				}
						else
						{
	                				Mag3D_Nhood[p][q][r] = 0.0;
	                				Phase3D_Nhood[p][q][r] = 0.0;
                    					BDFlag_3D[p][q][r] = false;
						}
					}
				}
		 		else
				{
         				for (r = 0; r < NHOOD_X_MAXDIM; r++){
	                			Mag3D_Nhood[p][q][r] = 0.0;
	                			Phase3D_Nhood[p][q][r] = 0.0;
                    				BDFlag_3D[p][q][r] = false;
					}
				}
                	}
		}
		else
        	{ 
			for (q = 0; q < NHOOD_Y_MAXDIM; q++){
				for (r = 0; r < NHOOD_X_MAXDIM; r++){
	              			Mag3D_Nhood[p][q][r] = 0.0;
	              			Phase3D_Nhood[p][q][r] = 0.0;
                   			BDFlag_3D[p][q][r] = false;
				}
			}
               }
	}

        Mag3D_Nhood[(NHOOD_Y_MAXDIM-1)/2][(NHOOD_X_MAXDIM-1)/2][(NHOOD_Z_MAXDIM-1)/2] = 0.0;
        Phase3D_Nhood[(NHOOD_Y_MAXDIM-1)/2][(NHOOD_X_MAXDIM-1)/2][(NHOOD_Z_MAXDIM-1)/2] = 0.0;
        VMag = ScannedObjectPtr->MagObject[i_new][slice+1][j_new][k_new]; /*Store the present value of the voxel*/
        VPhase = ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new]; /*Store the present value of the voxel*/

#ifdef ZERO_SKIPPING
			  /*Zero Skipping Algorithm*/
			 ZSFlag = true;
			 if(VMag == 0.0 && VPhase == 0.0 && Iter > 1) /*Iteration starts from 1. Iteration 0 corresponds to initial cost before ICD*/
			  {
					if (MagTime_Nhood[0] > 0.0 || MagTime_Nhood[1] > 0.0 || PhaseTime_Nhood[0] > 0.0 || PhaseTime_Nhood[1] > 0.0)
						ZSFlag = false;
			
					for(p = 0; p < NHOOD_Y_MAXDIM; p++)
						for(q = 0; q < NHOOD_X_MAXDIM; q++)
					  		for(r = 0; r < NHOOD_Z_MAXDIM; r++)
							  	if(Mag3D_Nhood[p][q][r] > 0.0 || Phase3D_Nhood[p][q][r] > 0.0)
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
		compute_voxel_update_Atten (SinogramPtr, ScannedObjectPtr, TomoInputsPtr, MagErrorSino, PhaseErrorSino, AMatrixPtr, /*VoxelLineResponse,*/ Mag3D_Nhood, Phase3D_Nhood, MagTime_Nhood, PhaseTime_Nhood, BDFlag_3D, BDFlag_Time, i_new, slice, j_new, k_new);
	    	MagUpdateMap[j_new][k_new] += sqrt(pow(ScannedObjectPtr->MagObject[i_new][slice+1][j_new][k_new] - VMag,2)+pow(ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new] - VPhase,2));
	    	total_vox_mag += sqrt(pow(ScannedObjectPtr->MagObject[i_new][slice+1][j_new][k_new],2) + pow(ScannedObjectPtr->PhaseObject[i_new][slice+1][j_new][k_new],2));
 	}
		else
		    (*zero_count)++;
       }
       }
       }
}

    
     for (p=0; p<maxview; p++)
     {
     	free(AMatrixPtr[p].values);
     	free(AMatrixPtr[p].index);
     }
     free(AMatrixPtr);
      return (total_vox_mag);
}


Real_t updateVoxels (int32_t time_begin, int32_t time_end, int32_t slice_begin, int32_t slice_end, int32_t xy_begin, int32_t xy_end, int32_t* x_rand_select, int32_t* y_rand_select, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** MagErrorSino, Real_arr_t*** PhaseErrorSino, Real_arr_t** DetectorResponse_XY, int32_t Iter, long int *zero_count, Real_arr_t** MagUpdateMap, uint8_t** Mask)
{
	Real_t total_vox_mag;
	total_vox_mag = updateVoxels_Atten  (time_begin, time_end, slice_begin, slice_end, xy_begin, xy_end, x_rand_select, y_rand_select, SinogramPtr, ScannedObjectPtr, TomoInputsPtr, MagErrorSino, PhaseErrorSino, DetectorResponse_XY, Iter, zero_count, MagUpdateMap, Mask);
	return (total_vox_mag);
}

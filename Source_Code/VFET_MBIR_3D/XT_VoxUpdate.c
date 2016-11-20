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

void compute_voxel_update_Atten (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, Real_arr_t*** ErrorSino_Unflip_x, Real_arr_t*** ErrorSino_Flip_x, Real_arr_t*** ErrorSino_Unflip_y, Real_arr_t*** ErrorSino_Flip_y, AMatrixCol* AMatrixPtr_X, AMatrixCol* AMatrixPtr_Y, Real_t MagPrior[3], Real_t ElecPrior, int32_t slice, int32_t j_new, int32_t k_new)
{
  	int32_t p, q, r, z_overlap_num;
	Real_t VMag[3], THETA1Mag[3], THETA2Mag[3][3], ProjectionEntry, ProjEntryComp, THETA1Elec, THETA2Elec;
  	int32_t i_r, i_t;

        VMag[0] = ObjPtr->MagPotentials[slice][j_new][k_new][0]; /*Store the present value of the voxel*/
        VMag[1] = ObjPtr->MagPotentials[slice][j_new][k_new][1]; /*Store the present value of the voxel*/
        VMag[2] = ObjPtr->MagPotentials[slice][j_new][k_new][2]; /*Store the present value of the voxel*/

	z_overlap_num = SinoPtr->z_overlap_num;

	THETA1Mag[0] = 0.0;
	THETA1Mag[1] = 0.0;
	THETA1Mag[2] = 0.0;

	THETA2Mag[0][0] = 0.0; THETA2Mag[0][1] = 0.0; THETA2Mag[0][2] = 0.0;
	THETA2Mag[1][0] = 0.0; THETA2Mag[1][1] = 0.0; THETA2Mag[1][2] = 0.0;
	THETA2Mag[2][0] = 0.0; THETA2Mag[2][1] = 0.0; THETA2Mag[2][2] = 0.0;

	for (p = 0; p < SinoPtr->Nx_p; p++){
		for (q = 0; q < AMatrixPtr_X[p].count; q++)
		{
      		    	i_r = (AMatrixPtr_X[p].index[q]);
       		    	ProjectionEntry = (AMatrixPtr_X[p].values[q]);
/*      	 	ProjectionEntry = (AMatrixPtr[p].values[q]);*/
			for (r = 0; r < ObjPtr->VoxelLineResp_X[k_new].count; r++)
			{ 
				i_t = ObjPtr->VoxelLineResp_X[k_new].index[r];
				THETA2Mag[0][1] += -ProjectionEntry*ProjectionEntry*SinoPtr->sine_x[p]*SinoPtr->cosine_x[p]*InpPtr->Weight*InpPtr->MagPhaseMultiple*InpPtr->MagPhaseMultiple;
					
				ProjEntryComp = ProjectionEntry*SinoPtr->cosine_x[p]*InpPtr->MagPhaseMultiple;
	        	   	THETA2Mag[0][0] += ProjEntryComp*ProjEntryComp*InpPtr->Weight;
               			THETA1Mag[0] += -((ErrorSino_Unflip_x[p][i_r][i_t])*ProjEntryComp*InpPtr->Weight);
				
				ProjEntryComp = -ProjectionEntry*SinoPtr->sine_x[p]*InpPtr->MagPhaseMultiple;
				THETA2Mag[1][1] += ProjEntryComp*ProjEntryComp*InpPtr->Weight;
               			THETA1Mag[1] += -((ErrorSino_Unflip_x[p][i_r][i_t])*ProjEntryComp*InpPtr->Weight);
            		}
		}
        }
	THETA2Mag[1][0] = THETA2Mag[0][1];
	
	for (p = 0; p < SinoPtr->Ny_p; p++){
		for (q = 0; q < AMatrixPtr_Y[p].count; q++)
		{
      		    	i_r = (AMatrixPtr_Y[p].index[q]);
       		    	ProjectionEntry = (AMatrixPtr_Y[p].values[q]);
/*      	 	ProjectionEntry = (AMatrixPtr[p].values[q]);*/
			for (r = 0; r < ObjPtr->VoxelLineResp_Y[j_new].count; r++)
			{ 
				i_t = ObjPtr->VoxelLineResp_Y[j_new].index[r];
				THETA2Mag[0][2] += -ProjectionEntry*ProjectionEntry*SinoPtr->sine_y[p]*SinoPtr->cosine_y[p]*InpPtr->Weight*InpPtr->MagPhaseMultiple*InpPtr->MagPhaseMultiple;
					
				ProjEntryComp = ProjectionEntry*SinoPtr->cosine_y[p]*InpPtr->MagPhaseMultiple;
	        	   	THETA2Mag[0][0] += ProjEntryComp*ProjEntryComp*InpPtr->Weight;
               			THETA1Mag[0] += -((ErrorSino_Unflip_y[p][i_r][i_t])*ProjEntryComp*InpPtr->Weight);
				
				ProjEntryComp = -ProjectionEntry*SinoPtr->sine_y[p]*InpPtr->MagPhaseMultiple;
				THETA2Mag[2][2] += ProjEntryComp*ProjEntryComp*InpPtr->Weight;
               			THETA1Mag[2] += -((ErrorSino_Unflip_y[p][i_r][i_t])*ProjEntryComp*InpPtr->Weight);
				
            		}
		}
        }
	THETA2Mag[2][0] = THETA2Mag[0][2];

            /*Solve the 1-D optimization problem
            TODO : What if theta1 = 0 ? Then this will give error*/

	MagFunctionalSubstitutionConstPrior(&(ObjPtr->MagPotentials[slice][j_new][k_new][0]), THETA1Mag, THETA2Mag, ObjPtr, InpPtr, MagPrior);
	
	for (p = 0; p < SinoPtr->Nx_p; p++){
		for (q = 0; q < AMatrixPtr_X[p].count; q++)
        	{
               	    	i_r = (AMatrixPtr_X[p].index[q]);
        	    	ProjectionEntry = (AMatrixPtr_X[p].values[q]);
        	    	/*ProjectionEntry = (AMatrixPtr[p].values[q]);*/
			for (r = 0; r < ObjPtr->VoxelLineResp_X[k_new].count; r++)
			{ 
				i_t = ObjPtr->VoxelLineResp_X[k_new].index[r];
	     /*   		ErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*VoxelLineResponse[slice].values[r]*(ObjPtr->Object[i_new][slice+1][j_new][k_new] - V));*/
				ProjEntryComp = ProjectionEntry*SinoPtr->cosine_x[p]*InpPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_x[p][i_r][i_t] -= (ProjEntryComp*(ObjPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]));

				ProjEntryComp = -ProjectionEntry*SinoPtr->sine_x[p]*InpPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_x[p][i_r][i_t] -= (ProjEntryComp*(ObjPtr->MagPotentials[slice][j_new][k_new][1] - VMag[1]));
				
	   		}
		}
	}
	
	for (p = 0; p < SinoPtr->Ny_p; p++){
		for (q = 0; q < AMatrixPtr_Y[p].count; q++)
        	{
               	    	i_r = (AMatrixPtr_Y[p].index[q]);
        	    	ProjectionEntry = (AMatrixPtr_Y[p].values[q]);
        	    	/*ProjectionEntry = (AMatrixPtr[p].values[q]);*/
			for (r = 0; r < ObjPtr->VoxelLineResp_Y[j_new].count; r++)
			{ 
				i_t = ObjPtr->VoxelLineResp_Y[j_new].index[r];
/*	        		ErrorSino[sino_view][i_r][i_t] -= (ProjectionEntry*VoxelLineResponse[slice].values[r]*(ObjPtr->Object[i_new][slice+1][j_new][k_new] - V));*/
				ProjEntryComp = ProjectionEntry*SinoPtr->cosine_y[p]*InpPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_y[p][i_r][i_t] -= (ProjEntryComp*(ObjPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]));
			
				ProjEntryComp = -ProjectionEntry*SinoPtr->sine_y[p]*InpPtr->MagPhaseMultiple;
	        		ErrorSino_Unflip_y[p][i_r][i_t] -= (ProjEntryComp*(ObjPtr->MagPotentials[slice][j_new][k_new][2] - VMag[2]));
	
	   		}
		}
	}
	
	ObjPtr->ErrorPotMag[slice][j_new][k_new][0] += (ObjPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0]); 
	ObjPtr->ErrorPotMag[slice][j_new][k_new][1] += (ObjPtr->MagPotentials[slice][j_new][k_new][1] - VMag[1]); 
	ObjPtr->ErrorPotMag[slice][j_new][k_new][2] += (ObjPtr->MagPotentials[slice][j_new][k_new][2] - VMag[2]); 
}

Real_t updateVoxels (int32_t xyz_begin, int32_t xyz_end, int32_t* x_rand_select, int32_t* y_rand_select, int32_t* z_rand_select, Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, Real_arr_t*** ErrorSino_Unflip_x, Real_arr_t*** ErrorSino_Flip_x, Real_arr_t*** ErrorSino_Unflip_y, Real_arr_t*** ErrorSino_Flip_y, Real_arr_t** DetectorResponse_X, Real_arr_t** DetectorResponse_Y, int32_t Iter, Real_t ***MagUpdateMap, Real_t *MagPotUpdate, Real_t *MagPotSum, uint8_t** Mask)
{
  int32_t p,slice,j_new,k_new,index_xyz;
  Real_t VMag[3], MagPrior[3]; /*numavg_x, numavg_y;*/
  Real_t total_vox_mag = 0.0;
  Real_t VElec, ElecPrior, detdist_r;

   /*printf ("maxview = %d, size of AMatrixCol = %d\n",maxview,sizeof(AMatrixCol));*/
  AMatrixCol* AMatrixPtr_X = (AMatrixCol*)get_spc(SinoPtr->Nx_p, sizeof(AMatrixCol));
  AMatrixCol* AMatrixPtr_Y = (AMatrixCol*)get_spc(SinoPtr->Ny_p, sizeof(AMatrixCol));
  uint8_t AvgNumXElements = (uint8_t)ceil(3*ObjPtr->delta_x/SinoPtr->delta_r);
  uint8_t AvgNumYElements = (uint8_t)ceil(3*ObjPtr->delta_y/SinoPtr->delta_r);
  
  for (p = 0; p < SinoPtr->Nx_p; p++)
  {
  	AMatrixPtr_X[p].values = (Real_t*)get_spc(AvgNumXElements,sizeof(Real_t));
  	AMatrixPtr_X[p].index  = (int32_t*)get_spc(AvgNumXElements,sizeof(int32_t));
  }

  for (p = 0; p < SinoPtr->Ny_p; p++)
  {
  	AMatrixPtr_Y[p].values = (Real_t*)get_spc(AvgNumYElements,sizeof(Real_t));
  	AMatrixPtr_Y[p].index  = (int32_t*)get_spc(AvgNumYElements,sizeof(int32_t));
  }  

      for (index_xyz = xyz_begin; index_xyz <= xyz_end; index_xyz++) 
      {
    /*    printf ("Entering index\n");*/ 
	k_new = x_rand_select[index_xyz];
        j_new = y_rand_select[index_xyz];
        slice = z_rand_select[index_xyz];
    	MagUpdateMap[slice][j_new][k_new] = 0;  
           /*printf ("Entering mask\n"); */
	  /*for (p = 0; p < SinoPtr->N_p; p++)
    	  {
		calcAMatrixColumnforAngle(SinoPtr, ObjPtr, DetectorResponse_XY, &(AMatrixPtr_X[p]), j_new, k_new, p);
    	  }*/
		/*numavg_x  = 0; numavg_y = 0;*/
	  	for (p = 0; p < SinoPtr->Nx_p; p++)
    	  	{
			/*calcAMatrixColumnforAngle(SinoPtr, ObjPtr, DetectorResponse_XY, &(AMatrixPtr_Y[p]), j_new, slice, p);*/
			detdist_r = (ObjPtr->y0 + ((Real_t)j_new+0.5)*ObjPtr->delta_y)*SinoPtr->cosine_x[p];
			detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_x[p];
			calcAMatrixColumnforAngle(SinoPtr, ObjPtr, DetectorResponse_X[p], &(AMatrixPtr_X[p]), detdist_r);
	/*		numavg_x += (Real_t)AMatrixPtr_X[p].count;*/
		}	
	  	
		for (p = 0; p < SinoPtr->Ny_p; p++)
		{
			detdist_r = (ObjPtr->x0 + ((Real_t)k_new+0.5)*ObjPtr->delta_x)*SinoPtr->cosine_y[p];
			detdist_r += -(ObjPtr->z0 + ((Real_t)slice+0.5)*ObjPtr->delta_z)*SinoPtr->sine_y[p];
          		calcAMatrixColumnforAngle(SinoPtr, ObjPtr, DetectorResponse_Y[p], &(AMatrixPtr_Y[p]), detdist_r);
	/*		numavg_y += (Real_t)AMatrixPtr_Y[p].count;*/

		}
/*		printf("slice = %d, j_new = %d, k_new = %d, numavg_x = %f, numavg_y = %f\n", slice, j_new, k_new, numavg_x/SinoPtr->N_p, numavg_y/SinoPtr->N_p);*/
        /*  	printf ("Entering slice\n");*/ 
            /*For a given (i,j,k) store its 26 point neighborhood*/           
	    /*if (Mask[j_new][k_new] == 1)
	    { */  
		MagPrior[0] = -(ObjPtr->ErrorPotMag[slice][j_new][k_new][0] - ObjPtr->MagPotentials[slice][j_new][k_new][0]);
		MagPrior[1] = -(ObjPtr->ErrorPotMag[slice][j_new][k_new][1] - ObjPtr->MagPotentials[slice][j_new][k_new][1]);
		MagPrior[2] = -(ObjPtr->ErrorPotMag[slice][j_new][k_new][2] - ObjPtr->MagPotentials[slice][j_new][k_new][2]); 
        
		VMag[0] = ObjPtr->MagPotentials[slice][j_new][k_new][0]; /*Store the present value of the voxel*/
        	VMag[1] = ObjPtr->MagPotentials[slice][j_new][k_new][1]; /*Store the present value of the voxel*/
        	VMag[2] = ObjPtr->MagPotentials[slice][j_new][k_new][2]; /*Store the present value of the voxel*/

		compute_voxel_update_Atten (SinoPtr, ObjPtr, InpPtr, ErrorSino_Unflip_x, ErrorSino_Flip_x, ErrorSino_Unflip_y, ErrorSino_Flip_y, AMatrixPtr_X, AMatrixPtr_Y, MagPrior, ElecPrior, slice, j_new, k_new);

		(MagUpdateMap[slice][j_new][k_new]) = sqrt(pow(ObjPtr->MagPotentials[slice][j_new][k_new][0] - VMag[0],2) + pow(ObjPtr->MagPotentials[slice][j_new][k_new][1] - VMag[1],2) + pow(ObjPtr->MagPotentials[slice][j_new][k_new][2] - VMag[2],2));
		(*MagPotUpdate) += (MagUpdateMap[slice][j_new][k_new]*MagUpdateMap[slice][j_new][k_new]); 
		(*MagPotSum) += (pow(ObjPtr->MagPotentials[slice][j_new][k_new][0],2) + pow(ObjPtr->MagPotentials[slice][j_new][k_new][1],2) + pow(ObjPtr->MagPotentials[slice][j_new][k_new][2],2)); 

       	     /*}*/
       }

    
     for (p=0; p<SinoPtr->Nx_p; p++)
     {
     	free(AMatrixPtr_X[p].values);
     	free(AMatrixPtr_X[p].index);
     }
     for (p=0; p<SinoPtr->Ny_p; p++)
     {
     	free(AMatrixPtr_Y[p].values);
     	free(AMatrixPtr_Y[p].index);
     }
     free(AMatrixPtr_X);
     free(AMatrixPtr_Y);
      return (total_vox_mag);
}



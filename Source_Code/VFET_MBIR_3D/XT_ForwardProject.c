#include <stdio.h>
#include "XT_Structures.h"
#include "XT_AMatrix.h"
#include "allocate.h"


void forward_project_voxel (Sinogram* SinogramPtr, Real_t* mag_voxel_val, Real_t elec_voxel_val, Real_arr_t*** ErrorSino_Unflip_z, Real_arr_t*** ErrorSino_Flip_z, AMatrixCol* AMatrixPtr/*, AMatrixCol* VoxelLineResponse*/, int32_t sino_idx, int32_t slice)
{
	int32_t m, idx, n, z_overlap_num;
	Real_t val, voxel_unflip, voxel_flip;
 
	voxel_unflip = mag_voxel_val[0]*SinogramPtr->cosine[sino_idx] - mag_voxel_val[1]*SinogramPtr->sine[sino_idx] + elec_voxel_val;
	voxel_flip = -mag_voxel_val[0]*SinogramPtr->cosine[sino_idx] + mag_voxel_val[1]*SinogramPtr->sine[sino_idx] + elec_voxel_val;
	z_overlap_num = SinogramPtr->z_overlap_num;
	for (m = 0; m < AMatrixPtr->count; m++)
	{
		idx = AMatrixPtr->index[m];
		val = AMatrixPtr->values[m];
		/*val = AMatrixPtr->values[m];
		for (n = 0; n < VoxelLineResponse->count; n++){*/
		for (n = 0; n < z_overlap_num; n++){
			/*ErrorSino[sino_idx][idx][VoxelLineResponse->index[n]] += voxel_val*val*VoxelLineResponse->values[n];*/
			ErrorSino_Unflip_z[sino_idx][idx][slice*z_overlap_num + n] += voxel_unflip*val;
			ErrorSino_Flip_z[sino_idx][idx][slice*z_overlap_num + n] += voxel_flip*val;
		}
	}
}


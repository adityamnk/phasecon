#include <stdio.h>
#include "XT_Structures.h"
#include "XT_AMatrix.h"
#include "allocate.h"

void mag_forward_project_voxel (Sinogram* SinogramPtr, TomoInputs* TomoInputsPtr, Real_t mag_voxel_val_par, Real_t mag_voxel_val_perp, Real_arr_t*** ErrorSino_Unflip_z, Real_arr_t*** ErrorSino_Flip_z, AMatrixCol* AMatrixPtr, AMatrixCol* VoxelLineResponse, int32_t sino_idx, Real_t cosine, Real_t sine)
{
	int32_t m, idx, n, z_overlap_num;
	Real_t val, voxel_unflip, voxel_flip;
 
	voxel_unflip = (mag_voxel_val_par*cosine - mag_voxel_val_perp*sine)*TomoInputsPtr->MagPhaseMultiple;
	z_overlap_num = SinogramPtr->z_overlap_num;
	for (m = 0; m < AMatrixPtr->count; m++)
	{
		idx = AMatrixPtr->index[m];
		val = AMatrixPtr->values[m];
		for (n = 0; n < VoxelLineResponse->count; n++){
			ErrorSino_Unflip_z[sino_idx][idx][VoxelLineResponse->index[n]] += voxel_unflip*val;
		}
	}
}



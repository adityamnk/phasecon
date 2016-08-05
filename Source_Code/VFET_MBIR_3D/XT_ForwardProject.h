#ifndef XT_FORWARDPROJECT_H

#define XT_FORWARDPROJECT_H
void mag_forward_project_voxel (Sinogram* SinogramPtr, TomoInputs* TomoInputsPtr, Real_t mag_voxel_val_par, Real_t mag_voxel_val_perp, Real_arr_t*** ErrorSino_Unflip_z, Real_arr_t*** ErrorSino_Flip_z, AMatrixCol* AMatrixPtr, AMatrixCol* VoxelLineResponse, int32_t sino_idx, Real_t cosine, Real_t sine);

#endif

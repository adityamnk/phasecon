

/* ============================================================================
 * Copyright (c) 2015 K. Aditya Mohan (Purdue University)
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


#include <stdio.h>
#include "XT_Structures.h"
#include "XT_AMatrix.h"
#include "allocate.h"



void forward_project_voxel_AMat1D (Sinogram* SinogramPtr, Real_t voxel_val, Real_arr_t*** ErrorSino, AMatrixCol* AMatrixPtr/*, AMatrixCol* VoxelLineResponse*/, int32_t sino_idx, int32_t slice)
{
	int32_t m, idx, n, z_overlap_num;
	Real_t val;
 
	z_overlap_num = SinogramPtr->z_overlap_num;
	for (m = 0; m < AMatrixPtr->count; m++)
	{
		idx = AMatrixPtr->index[m];
		val = AMatrixPtr->values[m];
		/*val = AMatrixPtr->values[m];
		for (n = 0; n < VoxelLineResponse->count; n++){*/
		for (n = 0; n < z_overlap_num; n++){
			/*ErrorSino[sino_idx][idx][VoxelLineResponse->index[n]] += voxel_val*val*VoxelLineResponse->values[n];*/
			ErrorSino[sino_idx][idx][slice*z_overlap_num + n] += voxel_val*val;
		}
	}
}

	
void forward_project_voxel_AMat2D (Sinogram* SinogramPtr, Real_t voxel_val, Real_arr_t*** ErrorSino, AMatrixCol* AMatrixPtr/*, AMatrixCol* VoxelLineResponse*/, int32_t sino_idx, int32_t slice)
{
	int32_t m, n;
	int32_t r_ax_start, r_ax_num, t_ax_start, t_ax_num;
	Real_t **AMatrix2D, *AMatrix2DLine;	

	t_ax_start = slice*SinogramPtr->z_overlap_num;
	t_ax_num = SinogramPtr->z_overlap_num;
	compute_2DAMatrixLine(SinogramPtr, &(AMatrix2DLine), AMatrixPtr, &r_ax_start, &r_ax_num);
	compute_LapMatrix_4m_AMatrix(SinogramPtr, &(AMatrix2D), &(AMatrix2DLine), &(r_ax_start), &(r_ax_num), &(t_ax_start), &(t_ax_num));
 
	for (m = 0; m < r_ax_num; m++)
	{
		for (n = 0; n < t_ax_num; n++){
			ErrorSino[sino_idx][r_ax_start+m][t_ax_start+n] += voxel_val*AMatrix2D[m][n];
		}
	}
	
	if (r_ax_num != 0 && t_ax_num != 0)
		multifree(AMatrix2D,2);
}

	
void forward_project_voxel (Sinogram* SinogramPtr, Real_t voxel_val, Real_arr_t*** ErrorSino, AMatrixCol* AMatrixPtr, /*AMatrixCol* VoxelLineResponse,*/ int32_t sino_idx, int32_t slice)
{
#ifdef PHASE_CONTRAST_TOMOGRAPHY
	forward_project_voxel_AMat2D (SinogramPtr, voxel_val, ErrorSino, AMatrixPtr, /*VoxelLineResponse,*/ sino_idx, slice);
#else
	forward_project_voxel_AMat1D (SinogramPtr, voxel_val, ErrorSino, AMatrixPtr, /*VoxelLineResponse,*/ sino_idx, slice);
#endif
}

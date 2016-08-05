/* ============================================================================
 * Copyright (c) 2013 K. Aditya Mohan (Purdue University)
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


#ifndef XT_INIT_H
#define XT_INIT_H
int32_t initStructures (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, int32_t mult_idx, int32_t mult_xyz[], float *data_unflip_x, float *data_unflip_y, float *proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, int32_t x_widnum, int32_t y_widnum, int32_t z_widnum, Real_t vox_wid, Real_t qggmrf_sigma, Real_t qggmrf_c, Real_t convg_thresh, Real_t admm_mu, int32_t admm_maxiters);
void argsParser (int argc, char **argv, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
void freeMemory(Sinogram* SinogramPtr, ScannedObject *ScannedObjectPtr, TomoInputs* TomoInputsPtr, FFTStruct* fftptr);
int32_t initPhantomStructures (Sinogram* SinoPtr, ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, float *proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, Real_t vox_wid);
void freePhantomMemory(Sinogram* SinogramPtr, ScannedObject *ScannedObjectPtr, TomoInputs* TomoInputsPtr, FFTStruct* fftptr);

#endif /*#ifndef XT_INIT_H*/

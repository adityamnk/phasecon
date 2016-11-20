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



#ifndef XT_STRUCTURES_H
#define XT_STRUCTURES_H
#include <stdint.h>
#include "XT_Constants.h"
#include <stdbool.h>
#include <fftw3.h>
  
/*Structure to store a single column(A_i) of the A-matrix*/
  typedef struct
  {
      Real_t* values; /*Store the non zero entries*/
      uint8_t count; /*The number of non zero values present in the column*/
      int32_t *index; /*This maps each value to its location in the column.*/
  } AMatrixCol;

/*Structure 'Sinogram' contains the sinogram itself and also other parameters related to the sinogram and the detector*/
  typedef struct
  {
   Real_arr_t ***Data_Unflip_x; /*Stores the retrieved phase images at one rotation axis*/
   Real_arr_t ***Data_Flip_x; /*Stores the retrieved phase images when the object is flipped*/
   Real_arr_t ***Data_Unflip_y; /*Stores the retrieved phase images at a second perpendicular rotation axis*/
   Real_arr_t ***Data_Flip_y; /*Stores the retrieved phase images when the object is flipped along the 2nd axis*/

   Real_arr_t ***ErrorSino_Unflip_x;
   Real_arr_t ***ErrorSino_Flip_x; 
   Real_arr_t ***ErrorSino_Unflip_y;
   Real_arr_t ***ErrorSino_Flip_y; 

    Real_arr_t** DetectorResponse_x;/*response of the detector as a function of distance between the voxel center and center of detector element (for tilt across x-axis)*/
    Real_arr_t** DetectorResponse_y;/*response of the detector as a function of distance between the voxel center and center of detector element (for tilt across y-axis)*/
    Real_arr_t* ZLineResponse;
    int32_t N_r;/*Number of detector elements in r direction (parallel to x-axis)*/
    int32_t N_t;/*Number of detector elements in t direction to be reconstructed (parallel to z axis)*/
    int32_t Nx_p;/*Total number of projections used in reconstruction*/
    int32_t Ny_p;/*Total number of projections used in reconstruction*/
    int32_t total_t_slices;/*Total number of slices in t-direction*/
    Real_t delta_r;/*Distance between successive measurements along r (or detector pixel width along t)*/
    Real_t delta_t;/*Distance between successive measurements along t (or detector pixel width along t)*/
    Real_t R0,RMax;/*location of leftmost and rightmost corner of detector along r*/
    Real_t T0,TMax;/*location of leftmost and rightmost corner of detector along t*/
    Real_t *cosine_x, *sine_x; /*stores the cosines and sines of the angles at which projections are acquired*/
    Real_t *cosine_y, *sine_y; /*stores the cosines and sines of the angles at which projections are acquired*/
    Real_t Length_R; /*Length of the detector along the r-dimension*/
    Real_t Length_T; /*Length of the detector along the t-dimension*/
    Real_t OffsetR; /*increments of distance between the center of the voxel and the midpoint of the detector along r axis */
    Real_t OffsetT; /*increments of distance between the center of the voxel and the midpoint of the detector along t axis*/
    Real_arr_t *ViewPtr_x; /*contains the values of projection angles*/
    Real_arr_t *ViewPtr_y; /*contains the values of projection angles*/

    int32_t z_overlap_num;

  } Sinogram;

  typedef struct
  {
    Real_arr_t ****MagPotGndTruth;
    Real_arr_t ***ElecPotGndTruth;
	
    Real_arr_t ****MagPotentials;
    Real_arr_t ***ElecPotentials;
    
    Real_arr_t ****MagPotDual;
    Real_arr_t ***ElecPotDual;
    
    Real_arr_t ****Magnetization;
    Real_arr_t ***ChargeDensity;
   
   Real_arr_t ****ErrorPotMag;
   Real_arr_t ***ErrorPotElec;

    Real_t ****MagFilt;
    Real_t ****ElecFilt;
 
    Real_arr_t ***MagPotUpdateMap;

    Real_t Length_X;/*maximum possible length of the object along x*/
    Real_t Length_Y;/*max length of object along y*/
    Real_t Length_Z;/*max length of object along z*/
    int32_t N_x;/*Number of voxels in x direction*/
    int32_t N_z;/*Number of voxels in z direction*/
    int32_t N_y;/*Number of voxels in y direction*/
/*However, we assume the voxel is a cube and also the resolution is equal in all the 3 dimensions*/
    /*Coordinates of the left corner of the object along x, y and z*/
    Real_t x0;
    Real_t z0;
    Real_t y0;
    Real_t delta_x;/*Voxel size in the x direction*/
    Real_t delta_y;/*Voxel size in the y direction*/
    Real_t delta_z;/*Voxel size in the z direction*/
    Real_t mult_xyz;/*voxel size as a multiple of detector pixel size along r*/

/*However, at finest resolution, delta_x = delta_y = delta_z*/ 
    Real_t BeamWidth; /*Beamwidth of the detector response*/
    
    Real_t Mag_Sigma[3]; /*regularization parameter over space (over x-y slice). Its a parameter of qGGMRF prior model*/ 
    Real_t Mag_C[3]; /*parameter c of qGGMRF prior model*/
    
    Real_t Elec_Sigma; /*regularization parameter over space (over x-y slice). Its a parameter of qGGMRF prior model*/ 
    Real_t Elec_C; /*parameter c of qGGMRF prior model*/

   int32_t NHICD_Iterations; /*percentage of voxel lines selected in NHICD*/

   AMatrixCol *VoxelLineResp_X, *VoxelLineResp_Y;
   
 } ScannedObject;

  typedef struct
  {
   	fftw_complex **fftforw_magarr;
   	fftw_complex **fftback_magarr;
   	fftw_plan *fftforw_magplan;
   	fftw_plan *fftback_magplan;

   	fftw_complex *fftforw_elecarr;
   	fftw_complex *fftback_elecarr;
   	fftw_plan fftforw_elecplan;
   	fftw_plan fftback_elecplan;

	int32_t x_num, y_num, z_num;
	int32_t x0, y0, z0;
  } FFTStruct;

typedef struct
  {
    int32_t NumIter; /*Maximum number of iterations that the ICD can be run. Normally, ICD converges before completing all the iterations and exits*/
    Real_t StopThreshold; /*ICD exits after the average update of the voxels becomes less than this threshold. Its specified in units of HU.*/
    /*Real_t RotCenter;*/ /*Center of rotation of the object as measured on the detector in units of pixels*/ 
    Real_t radius_obj;	/*Radius of the object within which the voxels are updated*/
    
    Real_t Mag_Sigma_Q[3]; /*The parameter sigma_s raised to power of q*/
    Real_t Mag_Sigma_Q_P[3]; /*Parameter sigma_s raised to power of q-p*/
    
    Real_t Elec_Sigma_Q; /*The parameter sigma_s raised to power of q*/
    Real_t Elec_Sigma_Q_P; /*Parameter sigma_s raised to power of q-p*/
   
    Real_t Weight;    
    
    Real_t Spatial_Filter[3][3][3]; /*Filter is the weighting kernel used in the prior model*/

    Real_t MagPhaseMultiple; /*Scaling multiple for the phase from magnetic potential*/ 
    Real_t ElecPhaseMultiple; /*Scaling multiple for the phase from electrostatic potential*/
    
    Real_t alpha; /*Value of over-relaxation*/
    Real_t cost_thresh; /*Convergence threshold on cost*/
    uint8_t initICD; /*used to specify the method of initializing the object before ICD
    If 0 object is initialized to 0. If 1, object is initialized from bin file directly without interpolation.
    If 2 object is interpolated by a factor of 2 in x-y plane and then initialized.
    If 3 object is interpolated by a factor of 2 in x-y-z space and then initialized*/
    uint8_t Write2Tiff; /*If set, tiff files are written*/
    uint8_t no_NHICD; /*If set, reconstruction goes not use NHICD*/
    uint8_t WritePerIter; /*If set, object and projection offset are written to bin and tiff files after every ICD iteration*/

    int32_t* x_NHICD_select; 
	/*x_NHICD_select and y_NHICD_select as pair 
	determines the voxels lines which are updated in a iteration of NHICD*/
    int32_t* y_NHICD_select;
    int32_t* z_NHICD_select;

    int32_t* x_rand_select;
    int32_t* y_rand_select;
    int32_t* z_rand_select;

    int32_t UpdateSelectNum;
    int32_t NHICDSelectNum;

    int32_t node_num; /*Number of nodes used*/
    int32_t node_rank; /*Rank of node*/

    uint8_t initMagUpMap; /*if set, initializes the magnitude update map*/
    FILE *debug_file_ptr; /*ptr to debug.log file*/

    Real_t ErrorSino_Cost;
    Real_t Forward_Cost;
    Real_t Prior_Cost;

    int32_t num_threads;
    Real_t ADMM_mu;
    Real_t ADMM_mu_incfact;
    Real_t ADMM_thresh;

    int32_t Head_MaxIter; 
    int32_t DensUpdate_MaxIter;
   
    Real_t DensUpdate_thresh; 
    Real_t Head_threshold; 
  } TomoInputs;

#endif /*#define XT_STRUCTURES_H*/

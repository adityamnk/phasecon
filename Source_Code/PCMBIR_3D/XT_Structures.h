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

/*Structure 'Sinogram' contains the sinogram itself and also other parameters related to the sinogram and the detector*/
  typedef struct
  {
   Real_arr_t ***Measurements_real; /*Stores the measurements (photon count measurements, intensity measurements, etc.)*/
   Real_arr_t ***Measurements_imag; /*Stores the measurements (photon count measurements, intensity measurements, etc.)*/

   Real_arr_t **Freq_Window;

   Real_arr_t ***Omega_real;
   Real_arr_t ***Omega_imag;
   Real_arr_t ***D_real;
   Real_arr_t ***D_imag;
   fftw_complex **fftforw_arr;
   fftw_complex **fftback_arr;
   fftw_plan *fftforw_plan;
   fftw_plan *fftback_plan;

   Real_arr_t ***MagProj;
   Real_arr_t ***PhaseProj;
   Real_arr_t ***MagErrorSino; /*Error sinogram of the magnitude component*/
   Real_arr_t ***PhaseErrorSino; /*Error sinogram of the phase component*/
   Real_arr_t ****MagTomoAux; /*ADMM auxiliary vector for the tomography split*/
   Real_arr_t ****PhaseTomoAux; /*ADMM auxiliary vector for the tomography split*/
   Real_arr_t ***MagTomoDual; /*ADMM dual vector corresponding to the tomography split*/
   Real_arr_t ***PhaseTomoDual; /*ADMM dual vector corresponding to the tomography split*/
   Real_arr_t ***MagPRetAux; /*ADMM auxiliary vector for the phase retrieval split*/
   Real_arr_t ***PhasePRetAux; /*ADMM auxiliary vector for the phase retrieval split*/
   Real_arr_t ***MagPRetDual; /*ADMM dual vector corresponding to the phase retrieval split*/
   Real_arr_t ***PhasePRetDual; /*ADMM dual vector corresponding to the phase retrieval split*/

   Real_arr_t** DetectorResponse;/*response of the detector as a function of distance between the voxel center and center of detector element*/
   Real_arr_t* ZLineResponse;
    int32_t N_r;/*Number of detector elements in r direction (parallel to x-axis)*/
    int32_t N_t;/*Number of detector elements in t direction to be reconstructed (parallel to z axis)*/
    int32_t N_p;/*Total number of projections used in reconstruction*/
    int32_t total_t_slices;/*Total number of slices in t-direction*/
    Real_t delta_r;/*Distance between successive measurements along r (or detector pixel width along t)*/
    Real_t delta_t;/*Distance between successive measurements along t (or detector pixel width along t)*/
    Real_t R0,RMax;/*location of leftmost and rightmost corner of detector along r*/
    Real_t T0,TMax;/*location of leftmost and rightmost corner of detector along t*/
    Real_t *cosine, *sine; /*stores the cosines and sines of the angles at which projections are acquired*/
    Real_t Length_R; /*Length of the detector along the r-dimension*/
    Real_t Length_T; /*Length of the detector along the t-dimension*/
    Real_t OffsetR; /*increments of distance between the center of the voxel and the midpoint of the detector along r axis */
    Real_t OffsetT; /*increments of distance between the center of the voxel and the midpoint of the detector along t axis*/
    Real_arr_t *ViewPtr; /*contains the values of projection angles*/
    Real_arr_t *TimePtr; /*contains the corresponding times of projections*/

    int32_t z_overlap_num;
    Real_arr_t ***off_constraint;
    int32_t off_constraint_size;
    int32_t off_constraint_num;

	Real_t Obj2Det_Distance;
	Real_t Delta_Over_Beta;
	Real_t Light_Energy;
	Real_t Light_Wavelength;
	Real_t Light_Wavenumber;
	Real_t GaussWinSigma;
  } Sinogram;

  typedef struct
  {
    Real_arr_t ****MagObject; /*Stores the reconstructed object from magnitude part of the projection*/
    Real_arr_t ****PhaseObject; /*Stores the reconstructed object from the phase part of the projection*/
    Real_arr_t ***MagObjMin; /*Min of MagObject*/
    Real_arr_t ***MagObjMax; /*Max of MagObject*/
    Real_arr_t ***PhaseObjMin; /*Min of PhaseObject*/
    Real_arr_t ***PhaseObjMax; /*Max of PhaseObject*/
    Real_arr_t ****UpdateMap; /*Stores the reconstructed object*/
    Real_t Length_X;/*maximum possible length of the object along x*/
    Real_t Length_Y;/*max length of object along y*/
    Real_t Length_Z;/*max length of object along z*/
    int32_t N_x;/*Number of voxels in x direction*/
    int32_t N_z;/*Number of voxels in z direction*/
    int32_t N_y;/*Number of voxels in y direction*/
/*However, we assume the voxel is a cube and also the resolution is equal in all the 3 dimensions*/
    int32_t N_time;/*Number of time slices of the object*/
    /*Coordinates of the left corner of the object along x, y and z*/
    Real_t x0;
    Real_t z0;
    Real_t y0;
    Real_t delta_xy;/*Voxel size in the x-y direction*/
    Real_t delta_z;/*Voxel size in the z direction*/
    Real_t mult_xy;/*voxel size as a multiple of detector pixel size along r*/
    Real_t mult_z;/*Voxel size as a multiple of detector pixel size along t*/

/*However, at finest resolution, delta_x = delta_y = delta_z*/ 
    Real_t BeamWidth; /*Beamwidth of the detector response*/
    
    Real_t Mag_Sigma_S; /*regularization parameter over space (over x-y slice). Its a parameter of qGGMRF prior model*/ 
    Real_t Mag_Sigma_T; /*regularization parameter across time. Its a parameter of qGGMRF prior model*/ 
    Real_t Mag_C_S; /*parameter c of qGGMRF prior model*/
    Real_t Mag_C_T; /*parameter c of qGGMRF prior model*/
    
    Real_t Phase_Sigma_S; /*regularization parameter over space (over x-y slice). Its a parameter of qGGMRF prior model*/ 
    Real_t Phase_Sigma_T; /*regularization parameter across time. Its a parameter of qGGMRF prior model*/ 
    Real_t Phase_C_S; /*parameter c of qGGMRF prior model*/
    Real_t Phase_C_T; /*parameter c of qGGMRF prior model*/

    int32_t **ProjIdxPtr; /*Dictates the mapping of projection views to time slices*/
    int32_t *ProjNum; /*Number of projections assigned to each time slices*/

   int32_t NHICD_Iterations; /*percentage of voxel lines selected in NHICD*/
   Real_arr_t *recon_times; /*Time gap between time slices in reconstruction*/
   Real_t delta_recon;
   Real_t DecorrTran[2][2]; 
 } ScannedObject;

  /*Structure to store a single column(A_i) of the A-matrix*/
  typedef struct
  {
      Real_t* values; /*Store the non zero entries*/
      uint8_t count; /*The number of non zero values present in the column*/
      int32_t *index; /*This maps each value to its location in the column.*/
  } AMatrixCol;

typedef struct
  {
    int32_t NumIter; /*Maximum number of iterations that the ICD can be run. Normally, ICD converges before completing all the iterations and exits*/
    Real_t StopThreshold; /*ICD exits after the average update of the voxels becomes less than this threshold. Its specified in units of HU.*/
    Real_t RotCenter; /*Center of rotation of the object as measured on the detector in units of pixels*/ 
    Real_t radius_obj;	/*Radius of the object within which the voxels are updated*/
    
    Real_t Mag_Sigma_S_Q; /*The parameter sigma_s raised to power of q*/
    Real_t Mag_Sigma_T_Q; /*Parameter sigma_t raised to power of q*/
    Real_t Mag_Sigma_S_Q_P; /*Parameter sigma_s raised to power of q-p*/
    Real_t Mag_Sigma_T_Q_P; /*Parameter sigma_t raised to power of q-p*/
    
    Real_t Phase_Sigma_S_Q; /*The parameter sigma_s raised to power of q*/
    Real_t Phase_Sigma_T_Q; /*Parameter sigma_t raised to power of q*/
    Real_t Phase_Sigma_S_Q_P; /*Parameter sigma_s raised to power of q-p*/
    Real_t Phase_Sigma_T_Q_P; /*Parameter sigma_t raised to power of q-p*/
    
    Real_t Spatial_Filter[NHOOD_Z_MAXDIM][NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM]; /*Filter is the weighting kernel used in the prior model*/
    Real_t Time_Filter[(NHOOD_TIME_MAXDIM-1)/2]; /*Filter is the weighting kernel used in the prior model*/
    
    Real_arr_t*** Weight; /*Stores the weight matrix (noise matrix) values used in forward model*/
    
    Real_t alpha; /*Value of over-relaxation*/
    Real_t cost_thresh; /*Convergence threshold on cost*/
    uint8_t initICD; /*used to specify the method of initializing the object before ICD
    If 0 object is initialized to 0. If 1, object is initialized from bin file directly without interpolation.
    If 2 object is interpolated by a factor of 2 in x-y plane and then initialized.
    If 3 object is interpolated by a factor of 2 in x-y-z space and then initialized*/
    uint8_t Write2Tiff; /*If set, tiff files are written*/
    uint8_t no_NHICD; /*If set, reconstruction goes not use NHICD*/
    uint8_t WritePerIter; /*If set, object and projection offset are written to bin and tiff files after every ICD iteration*/
    int32_t num_z_blocks; /*z axis slices are split to num_z_blocks which are then used for multithreading*/
    int32_t prevnum_z_blocks; /*z axis slices are split to num_z_blocks which are then used for multithreading*/
    int32_t*** x_NHICD_select; 
	/*x_NHICD_select and y_NHICD_select as pair 
	determines the voxels lines which are updated in a iteration of NHICD*/
    int32_t*** y_NHICD_select;
    int32_t*** x_rand_select;
    int32_t*** y_rand_select;
    int32_t** UpdateSelectNum; /*Number of voxels selected for HICD updates*/
    int32_t** NHICDSelectNum; /*Number of voxels selected for NHICD updates*/

    int32_t node_num; /*Number of nodes used*/
    int32_t node_rank; /*Rank of node*/

    uint8_t initMagUpMap; /*if set, initializes the magnitude update map*/
    FILE *debug_file_ptr; /*ptr to debug.log file*/

    Real_t ErrorSino_Cost;
    Real_t Forward_Cost;
    Real_t Prior_Cost;

    int32_t num_threads;
    Real_t ADMM_mu;
    Real_t ADMM_nu;
    Real_t NMS_rho;
    Real_t NMS_chi;
    Real_t NMS_gamma;
    Real_t NMS_sigma;

    int32_t NMS_MaxIter;
    int32_t Head_MaxIter; 
    int32_t PRet_MaxIter;
    int32_t SteepDes_MaxIter;
    
    Real_t NMS_threshold;
    Real_t Head_threshold; 
    Real_t PRet_threshold;
    Real_t SteepDes_threshold;

    uint8_t recon_type;
  } TomoInputs;

#endif /*#define XT_STRUCTURES_H*/

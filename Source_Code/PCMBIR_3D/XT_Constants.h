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



#ifndef XT_CONSTANTS_H
#define XT_CONSTANTS_H

/*#define POSITIVITY_CONSTRAINT*/
#define EXTRA_DEBUG_MESSAGES
#include <float.h>

typedef double Real_t;
typedef double Real_arr_t; /*Don't change to 'double' without first changing the floats to doubles in XT_Engine.c*/
#define MPI_REAL_DATATYPE MPI_DOUBLE
#define MPI_REAL_ARR_DATATYPE MPI_DOUBLE
#define EPSILON_ERROR DBL_MIN
#define INFINITE_COST DBL_MAX

/*#define EXPECTED_COUNT_MEASUREMENT 35000*/ 
/*#define FAR_FIELD_DIFFRACTION*/
/*#define REF_IND_DEC_1 6.0078e-7 *//*Al*/ 
/*#define REF_IND_DEC_2 5.3682e-7 */ /*Si*/
/*#define REF_IND_DEC_2 0.0 *//*Al*/ 
/*#define REF_IND_DEC_1 0.0 *//*Al*/ 
/*#define ABSORP_COEF_1 7.6069e-10*/ /*Al*/
/*#define ABSORP_COEF_2 2.0397e-9*/ /*Al*/
/*#define ABSORP_COEF_2 8.7435e-10*/ /*Si*/
/*#define ABSORP_COEF_1 0.0*/ /*Si*/
/*#define ABSORP_COEF_2 0.0*/ /*Al*/

#define REF_IND_DEC_1 4.628e-07
#define REF_IND_DEC_2 7.789e-07
#define ABSORP_COEF_1 1.6937e-11
#define ABSORP_COEF_2 4.9250e-11

#define ABSORP_COEF_MIN ABSORP_COEF_2 - (ABSORP_COEF_1 - ABSORP_COEF_2)
#define ABSORP_COEF_MAX ABSORP_COEF_1 + (ABSORP_COEF_1 - ABSORP_COEF_2)/4.0
/*#define ABSORP_COEF_MIN ABSORP_COEF_1/2.0
#define ABSORP_COEF_MAX 2.0*ABSORP_COEF_2*/

#define REF_IND_DEC_MIN REF_IND_DEC_1 - (REF_IND_DEC_2 - REF_IND_DEC_1)
#define REF_IND_DEC_MAX REF_IND_DEC_2 + (REF_IND_DEC_2 - REF_IND_DEC_1)/4.0
/*#define REF_IND_DEC_MIN 0.0
#define REF_IND_DEC_MAX 1.5e-6*/

#define MAGOBJECT_INIT_VAL 0
#define PHASEOBJECT_INIT_VAL 0

#define ATT_COEF_1 (4*M_PI*ABSORP_COEF_1/LIGHT_WAVELENGTH)
#define ATT_COEF_2 (4*M_PI*ABSORP_COEF_2/LIGHT_WAVELENGTH)

#define PLANCKS_CONSTANT 4.135668e-19 /*Units of keV*s*/
#define LIGHT_SPEED 299792458e+6 /*Units of um/s*/

#define EXPECTED_COUNT_MEASUREMENT 10000
#define ZERO_SKIPPING
#define MAG_PHANTOM_FILEPATH "/scratch/conte/m/mohank/Sim_Datasets/Absorp_Cylinders_Phantom.bin"
#define PHASE_PHANTOM_FILEPATH "/scratch/conte/m/mohank/Sim_Datasets/RefIndex_Cylinders_Phantom.bin"
#define PHANTOM_SUPPORT_FILEPATH "/scratch/conte/m/mohank/Sim_Datasets/Cylinders_Phantom_Support.bin"
#define MIN_OBJ_FILEPATH "/scratch/rice/m/mohank/Sim_Datasets/Phantom3D_min.bin"
#define MAX_OBJ_FILEPATH "/scratch/rice/m/mohank/Sim_Datasets/Phantom3D_max.bin"
#define PROJ_LENGTH_FILEPATH "/scratch/rice/m/mohank/Sim_Datasets/ProjLength.bin"
#define PHANTOM_OFFSET 511
#define MEASUREMENTS_FILENAME "measurements"
#define WEIGHTS_FILENAME "weights"
#define BRIGHTS_FILENAME "brights"

#define MAGOBJECT_FILENAME "mag_object"
#define PHASEOBJECT_FILENAME "phase_object"
#define PHANTOM_MAGOBJECT_FILENAME "phan_mag_object"
#define PHANTOM_PHASEOBJECT_FILENAME "phan_phase_object"
#define INIT_MAGOBJECT_FILENAME "init_mag_object"
#define INIT_PHASEOBJECT_FILENAME "init_phase_object"
#define PAG_MAGOBJECT_FILENAME "pag_mag_object"
#define PAG_PHASEOBJECT_FILENAME "pag_phase_object"
#define MAGTOMOAUX_FILENAME "mag_tomo_aux"
#define PHASETOMOAUX_FILENAME "phase_tomo_aux"
#define PAG_MAGTOMOAUX_FILENAME "pag_mag_tomo_aux"
#define PAG_PHASETOMOAUX_FILENAME "pag_phase_tomo_aux"
#define PAG_MAGRET_FILENAME "pag_mag_ret"
#define PAG_PHASERET_FILENAME "pag_phase_ret"
#define MAGTOMODUAL_FILENAME "mag_tomo_dual"
#define PHASETOMODUAL_FILENAME "phase_tomo_dual"
#define MAGPRETAUX_FILENAME "mag_pret_aux"
#define PHASEPRETAUX_FILENAME "phase_pret_aux"
#define MAGPRETDUAL_FILENAME "mag_pret_dual"
#define PHASEPRETDUAL_FILENAME "phase_pret_dual"
#define OMEGAREAL_FILENAME "omega_real"
#define OMEGAIMAG_FILENAME "omega_imag"
#define OMEGAABS_FILENAME "omega_abs"

#define PROJ_OFFSET_FILENAME "proj_offset"
#define UPDATE_MAP_FILENAME "update_map"
#define RUN_STATUS_FILENAME "status"
#define VAR_PARAM_FILENAME "variance_estimate"
#define SCALED_MAGERROR_SINO_FILENAME "scaled_magerrorsino"
#define SCALED_PHASEERROR_SINO_FILENAME "scaled_phaseerrorsino"
#define DETECTOR_RESPONSE_FILENAME "detector_response"
#define PROJ_SELECT_FILENAME "proj_select"
#define COST_FILENAME "cost"
#define ORIG_COST_FILENAME "orig_cost"

#define MRF_P 1.2
#define MRF_Q 2.0

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

#ifndef M_PI_2
#define M_PI_2         1.57079632679489661923132169163975144   /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4         0.785398163397448309615660845819875721  /* pi/4 */
#endif

#define PROFILE_RESOLUTION 1536
#define BEAM_RESOLUTION 512
#define DETECTOR_RESPONSE_BINS 256

#define NHOOD_Z_MAXDIM 3
#define NHOOD_Y_MAXDIM 3
#define NHOOD_X_MAXDIM 3
#define NHOOD_TIME_MAXDIM 3

#define MAX_NUM_ITERATIONS 1000
#define OVER_RELAXATION_FACTOR 1.5
#define ENABLE_TIFF_WRITES 1 /*To disable generating tiff images use '0'*/
#define COST_CONVG_THRESHOLD 0.1
#define PROJ_OFFSET_INIT 0
#define NO_NHICD 0
#define WRITE_EVERY_ITER 1
#define ZINGER_ENABLE_PARAM_T 4.0
#define ZINGER_ENABLE_PARAM_DELTA 0.2
#define VAR_PARAM_INIT 1
#define COMPUTE_RMSE_CONVG 0
#define ZINGER_DISABLE_PARAM_T 100000
#define ZINGER_DISABLE_PARAM_DELTA 1
#define MIN_XY_RECON_RES 16
#define MAX_MULTRES_NUM 1
#define MIN_ROWS_PER_NODE 2
#define MIN_PROJECTION_ROWS 4

#define HFIELD_UNIT_CONV_CONST 0.0001
#define AIR_MASS_ATT_COEFF 0.496372335005353 /*in cm^2/g. computed using cubic interpolation*/
#define WATER_MASS_ATT_COEFF 0.521225397034623

#define WATER_DENSITY 1.0 /*in g/cm^3*/ 
#define AIR_DENSITY 0.001205
#define HOUNSFIELD_WATER_MAP 1000
#define HOUNSFIELD_AIR_MAP 0
#ifndef HOUNSFIELD_MAX
	#define HOUNSFIELD_MAX 60000
#endif
#ifndef HOUNSFIELD_MIN
	#define HOUNSFIELD_MIN 10000
#endif

#define PHANTOM_XY_SIZE 1024
#define PHANTOM_Z_SIZE 128

#define MAGTOMOAUX_FILENAME "mag_tomo_aux"
#define PHASETOMOAUX_FILENAME "phase_tomo_aux"
#define MAGTOMODUAL_FILENAME "mag_tomo_dual"
#define PHASETOMODUAL_FILENAME "phase_tomo_dual"
#define MAGPRETAUX_FILENAME "mag_pret_aux"
#define PHASEPRETAUX_FILENAME "phase_pret_aux"
#define MAGPRETDUAL_FILENAME "mag_pret_dual"
#define PHASEPRETDUAL_FILENAME "phase_pret_dual"
#define OMEGAREAL_FILENAME "omega_real"
#define OMEGAIMAG_FILENAME "omega_imag"
#define OMEGAABS_FILENAME "omega_abs"

#define PROJ_OFFSET_FILENAME "proj_offset"
#define UPDATE_MAP_FILENAME "update_map"
#define RUN_STATUS_FILENAME "status"
#define VAR_PARAM_FILENAME "variance_estimate"
#define SCALED_MAGERROR_SINO_FILENAME "scaled_magerrorsino"
#define SCALED_PHASEERROR_SINO_FILENAME "scaled_phaseerrorsino"
#define DETECTOR_RESPONSE_FILENAME "detector_response"
#define PROJ_SELECT_FILENAME "proj_select"
#define COST_FILENAME "cost"
#define ORIG_COST_FILENAME "orig_cost"

#endif /*#ifndef XT_CONSTANTS_H*/

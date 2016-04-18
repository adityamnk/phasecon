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
/*#define EXTRA_DEBUG_MESSAGES*/
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

/*#define INIT_GROUND_TRUTH_PHANTOM*/

#define PHANTOM_Z_SIZE 256
#define PHANTOM_XY_SIZE 256

#define REF_IND_DEC_1 4.628e-07
#define REF_IND_DEC_2 7.789e-07
#define ABSORP_COEF_1 1.6937e-11
#define ABSORP_COEF_2 4.9250e-11

#define ABSORP_COEF_MIN ABSORP_COEF_2 - (ABSORP_COEF_1 - ABSORP_COEF_2)
#define ABSORP_COEF_MAX ABSORP_COEF_1 + (ABSORP_COEF_1 - ABSORP_COEF_2)/4.0

#define REF_IND_DEC_MIN REF_IND_DEC_1 - (REF_IND_DEC_2 - REF_IND_DEC_1)
#define REF_IND_DEC_MAX REF_IND_DEC_2 + (REF_IND_DEC_2 - REF_IND_DEC_1)/4.0

#define MAGOBJECT_INIT_VAL 0
#define ELECOBJECT_INIT_VAL 0

#define EXPECTED_COUNT_MEASUREMENT 2500
#define ZERO_SKIPPING

#define DATA_UNFLIP_X_FILENAME "data_unflip_x"
#define DATA_FLIP_X_FILENAME "data_flip_x"
#define DATA_UNFLIP_Y_FILENAME "data_unflip_y"
#define DATA_FLIP_Y_FILENAME "data_flip_y"

#define ERRORSINO_UNFLIP_X_FILENAME "errorsino_unflip_x"
#define ERRORSINO_FLIP_X_FILENAME "errorsino_flip_x"
#define ERRORSINO_UNFLIP_Y_FILENAME "errorsino_unflip_y"
#define ERRORSINO_FLIP_Y_FILENAME "errorsino_flip_y"

#define WEIGHTS_FILENAME "weights"

#define MAGOBJECT_FILENAME "mag_object"
#define ELECOBJECT_FILENAME "elec_object"
#define PHANTOM_MAGOBJECT_FILENAME "MagVecPotPhantom"
#define PHANTOM_ELECOBJECT_FILENAME "ElecVecPotPhantom"
#define INIT_MAGOBJECT_FILENAME "init_mag_object"
#define INIT_ELECOBJECT_FILENAME "init_elec_object"

#define UPDATE_MAP_FILENAME "update_map"
#define RUN_STATUS_FILENAME "status"
#define DETECTOR_RESPONSE_FILENAME "detector_response"
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
#define BEAM_RESOLUTION 256
#define DETECTOR_RESPONSE_BINS 256

#define MAX_NUM_ITERATIONS 1000
#define OVER_RELAXATION_FACTOR 1.0
#define ENABLE_TIFF_WRITES 1 /*To disable generating tiff images use '0'*/
#define COST_CONVG_THRESHOLD 0.1
#define NO_NHICD 0
#define WRITE_EVERY_ITER 0
#define MIN_XY_RECON_RES 16
#define MAX_MULTRES_NUM 4
#define MIN_ROWS_PER_NODE 2
#define MIN_PROJECTION_ROWS 4

#define UPDATE_MAP_FILENAME "update_map"
#define RUN_STATUS_FILENAME "status"
#define DETECTOR_RESPONSE_FILENAME "detector_response"
#define COST_FILENAME "cost"
#define ORIG_COST_FILENAME "orig_cost"

#endif /*#ifndef XT_CONSTANTS_H*/

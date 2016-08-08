#include "XT_Constants.h"
#include <stdio.h>
#include "XT_Structures.h"
#include "XT_Prior.h"
#include "XT_AMatrix.h"
#include <math.h>
#include "allocate.h"

void cmplx_mult (Real_t* real, Real_t* imag, Real_t x, Real_t y, Real_t a, Real_t b)
{
	*real = x*a - y*b;
	*imag = x*b + y*a;
}

void compute_magcrossprodtran (Real_arr_t**** magobjinp, Real_arr_t**** magobjout, Real_arr_t**** magimp, FFTStruct *fftptr, int32_t N_z, int32_t N_y, int32_t N_x, Real_t tran_sign)
{
	int32_t i, j, k;
	int64_t idx, idx_i, idx_j, idx_k;
	Real_t real_xz, real_xy, real_yz, real_yx, real_zx, real_zy, imag_xz, imag_xy, imag_yz, imag_yx, imag_zx, imag_zy, N;
	N = fftptr->z_num*fftptr->y_num*fftptr->x_num;

	fftw_complex** fftforw_magarr = fftptr->fftforw_magarr;
	fftw_plan* fftforw_magplan = fftptr->fftforw_magplan;
	fftw_complex** fftback_magarr = fftptr->fftback_magarr;
	fftw_plan* fftback_magplan = fftptr->fftback_magplan;

	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;

		if (i >= fftptr->z0 && i < fftptr->z0 + N_z && j >= fftptr->y0 && j < fftptr->y0 + N_y && k >= fftptr->x0 && k < fftptr->x0 + N_x)
		{
			idx_i = i - fftptr->z0;
			idx_j = j - fftptr->y0;
			idx_k = k - fftptr->x0;
			
			fftforw_magarr[0][idx][0] = magobjinp[idx_i][idx_j][idx_k][0];
			fftforw_magarr[1][idx][0] = magobjinp[idx_i][idx_j][idx_k][1];
			fftforw_magarr[2][idx][0] = magobjinp[idx_i][idx_j][idx_k][2];
		
			fftforw_magarr[0][idx][1] = 0;
			fftforw_magarr[1][idx][1] = 0;
			fftforw_magarr[2][idx][1] = 0;
		}
		else
		{
			fftforw_magarr[0][idx][0] = 0;
			fftforw_magarr[1][idx][0] = 0;
			fftforw_magarr[2][idx][0] = 0;
		
			fftforw_magarr[0][idx][1] = 0;
			fftforw_magarr[1][idx][1] = 0;
			fftforw_magarr[2][idx][1] = 0;
		}
	}
	
	fftw_execute(fftforw_magplan[0]);
	fftw_execute(fftforw_magplan[1]);
	fftw_execute(fftforw_magplan[2]);

	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;

		cmplx_mult (&real_zx, &imag_zx, magimp[i][j][k][2*0+0], tran_sign*magimp[i][j][k][2*0+1], fftforw_magarr[2][idx][0], fftforw_magarr[2][idx][1]);
		cmplx_mult (&real_zy, &imag_zy, magimp[i][j][k][2*0+0], tran_sign*magimp[i][j][k][2*0+1], fftforw_magarr[1][idx][0], fftforw_magarr[1][idx][1]);
		
		cmplx_mult (&real_yx, &imag_yx, magimp[i][j][k][2*1+0], tran_sign*magimp[i][j][k][2*1+1], fftforw_magarr[2][idx][0], fftforw_magarr[2][idx][1]);
		cmplx_mult (&real_yz, &imag_yz, magimp[i][j][k][2*1+0], tran_sign*magimp[i][j][k][2*1+1], fftforw_magarr[0][idx][0], fftforw_magarr[0][idx][1]);
		
		cmplx_mult (&real_xy, &imag_xy, magimp[i][j][k][2*2+0], tran_sign*magimp[i][j][k][2*2+1], fftforw_magarr[1][idx][0], fftforw_magarr[1][idx][1]);
		cmplx_mult (&real_xz, &imag_xz, magimp[i][j][k][2*2+0], tran_sign*magimp[i][j][k][2*2+1], fftforw_magarr[0][idx][0], fftforw_magarr[0][idx][1]);
		
		fftback_magarr[2][idx][0] = (real_zy - real_yz);
		fftback_magarr[2][idx][1] = (imag_zy - imag_yz);
		
		fftback_magarr[1][idx][0] = (-real_zx + real_xz);
		fftback_magarr[1][idx][1] = (-imag_zx + imag_xz);

		fftback_magarr[0][idx][0] = (real_yx - real_xy);
		fftback_magarr[0][idx][1] = (imag_yx - imag_xy);
	}
	
	fftw_execute(fftback_magplan[0]);
	fftw_execute(fftback_magplan[1]);
	fftw_execute(fftback_magplan[2]);
	
	for (i = 0; i < N_z; i++) for (j = 0; j < N_y; j++) for (k = 0; k < N_x; k++)
	{
		idx = (i + fftptr->z0)*fftptr->y_num*fftptr->x_num + (j + fftptr->y0)*fftptr->x_num + (k + fftptr->x0);
		magobjout[i][j][k][2] = fftback_magarr[2][idx][0]/N;
		magobjout[i][j][k][1] = fftback_magarr[1][idx][0]/N;
		magobjout[i][j][k][0] = fftback_magarr[0][idx][0]/N;
	}
	 
}

void compute_magcrossprodstep (Real_arr_t**** magobjinp, Real_arr_t**** magimp, FFTStruct *fftptr, int32_t N_z, int32_t N_y, int32_t N_x, Real_t* alpha_mag)
{
	int32_t i, j, k;
	int64_t idx, idx_i, idx_j, idx_k;
	Real_t real_1, imag_1, real_2, imag_2, real_x, imag_x, real_y, imag_y, real_z, imag_z, N;
	
	fftw_complex** fftforw_magarr = fftptr->fftforw_magarr;
	fftw_plan* fftforw_magplan = fftptr->fftforw_magplan;
	fftw_complex** fftback_magarr = fftptr->fftback_magarr;
	fftw_plan* fftback_magplan = fftptr->fftback_magplan;

	N = fftptr->z_num*fftptr->y_num*fftptr->x_num;
	
	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;

		if (i >= fftptr->z0 && i < fftptr->z0 + N_z && j >= fftptr->y0 && j < fftptr->y0 + N_y && k >= fftptr->x0 && k < fftptr->x0 + N_x)
		{	
			idx_i = i - fftptr->z0;
			idx_j = j - fftptr->y0;
			idx_k = k - fftptr->x0;
			
			fftforw_magarr[0][idx][0] = magobjinp[idx_i][idx_j][idx_k][0];
			fftforw_magarr[1][idx][0] = magobjinp[idx_i][idx_j][idx_k][1];
			fftforw_magarr[2][idx][0] = magobjinp[idx_i][idx_j][idx_k][2];
		
			fftforw_magarr[0][idx][1] = 0;
			fftforw_magarr[1][idx][1] = 0;
			fftforw_magarr[2][idx][1] = 0;
		}
		else
		{
			fftforw_magarr[0][idx][0] = 0;
			fftforw_magarr[1][idx][0] = 0;
			fftforw_magarr[2][idx][0] = 0;
		
			fftforw_magarr[0][idx][1] = 0;
			fftforw_magarr[1][idx][1] = 0;
			fftforw_magarr[2][idx][1] = 0;
		}
	}
	
	fftw_execute(fftforw_magplan[0]);
	fftw_execute(fftforw_magplan[1]);
	fftw_execute(fftforw_magplan[2]);

	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;
		
		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*0+0], -magimp[i][j][k][2*0+1], magimp[i][j][k][2*0+0], magimp[i][j][k][2*0+1]);
		cmplx_mult (&real_2, &imag_2, magimp[i][j][k][2*1+0], -magimp[i][j][k][2*1+1], magimp[i][j][k][2*1+0], magimp[i][j][k][2*1+1]);
		cmplx_mult (&real_x, &imag_x, real_1 + real_2, imag_1 + imag_2, fftforw_magarr[2][idx][0], fftforw_magarr[2][idx][1]);
		
		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*1+0], -magimp[i][j][k][2*1+1], magimp[i][j][k][2*2+0], magimp[i][j][k][2*2+1]);
		cmplx_mult (&real_y, &imag_y, real_1, imag_1, fftforw_magarr[1][idx][0], fftforw_magarr[1][idx][1]);
		
		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*0+0], -magimp[i][j][k][2*0+1], magimp[i][j][k][2*2+0], magimp[i][j][k][2*2+1]);
		cmplx_mult (&real_z, &imag_z, real_1, imag_1, fftforw_magarr[0][idx][0], fftforw_magarr[0][idx][1]);
		
		fftback_magarr[2][idx][0] = real_x - real_y - real_z;
		fftback_magarr[2][idx][1] = imag_x - imag_y - imag_z;

		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*2+0], -magimp[i][j][k][2*2+1], magimp[i][j][k][2*1+0], magimp[i][j][k][2*1+1]);
		cmplx_mult (&real_x, &imag_x, real_1, imag_1, fftforw_magarr[2][idx][0], fftforw_magarr[2][idx][1]);
		
		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*0+0], -magimp[i][j][k][2*0+1], magimp[i][j][k][2*0+0], magimp[i][j][k][2*0+1]);
		cmplx_mult (&real_2, &imag_2, magimp[i][j][k][2*2+0], -magimp[i][j][k][2*2+1], magimp[i][j][k][2*2+0], magimp[i][j][k][2*2+1]);
		cmplx_mult (&real_y, &imag_y, real_1 + real_2, imag_1 + imag_2, fftforw_magarr[1][idx][0], fftforw_magarr[1][idx][1]);
		
		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*0+0], -magimp[i][j][k][2*0+1], magimp[i][j][k][2*1+0], magimp[i][j][k][2*1+1]);
		cmplx_mult (&real_z, &imag_z, real_1, imag_1, fftforw_magarr[0][idx][0], fftforw_magarr[0][idx][1]);
		
		fftback_magarr[1][idx][0] = -real_x + real_y - real_z;
		fftback_magarr[1][idx][1] = -imag_x + imag_y - imag_z;

		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*2+0], -magimp[i][j][k][2*2+1], magimp[i][j][k][2*0+0], magimp[i][j][k][2*0+1]);
		cmplx_mult (&real_x, &imag_x, real_1, imag_1, fftforw_magarr[2][idx][0], fftforw_magarr[2][idx][1]);
		
		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*1+0], -magimp[i][j][k][2*1+1], magimp[i][j][k][2*0+0], magimp[i][j][k][2*0+1]);
		cmplx_mult (&real_y, &imag_y, real_1, imag_1, fftforw_magarr[1][idx][0], fftforw_magarr[1][idx][1]);
		
		cmplx_mult (&real_1, &imag_1, magimp[i][j][k][2*1+0], -magimp[i][j][k][2*1+1], magimp[i][j][k][2*1+0], magimp[i][j][k][2*1+1]);
		cmplx_mult (&real_2, &imag_2, magimp[i][j][k][2*2+0], -magimp[i][j][k][2*2+1], magimp[i][j][k][2*2+0], magimp[i][j][k][2*2+1]);
		cmplx_mult (&real_z, &imag_z, real_1 + real_2, imag_1 + imag_2, fftforw_magarr[0][idx][0], fftforw_magarr[0][idx][1]);
		
		fftback_magarr[0][idx][0] = -real_x - real_y + real_z;
		fftback_magarr[0][idx][1] = -imag_x - imag_y + imag_z;

	}

	fftw_execute(fftback_magplan[0]);
	fftw_execute(fftback_magplan[1]);
	fftw_execute(fftback_magplan[2]);

	*alpha_mag = 0;
	for (i = 0; i < N_z; i++) for (j = 0; j < N_y; j++) for (k = 0; k < N_x; k++)
	{
		idx = (i + fftptr->z0)*fftptr->y_num*fftptr->x_num + (j + fftptr->y0)*fftptr->x_num + (k + fftptr->x0);
		
		*alpha_mag += magobjinp[i][j][k][2]*fftback_magarr[2][idx][0];	
		*alpha_mag += magobjinp[i][j][k][1]*fftback_magarr[1][idx][0];	
		*alpha_mag += magobjinp[i][j][k][0]*fftback_magarr[0][idx][0];	
	}
	*alpha_mag /= N;

	
}	


void compute_mag_gradstep(ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, Real_arr_t**** grad_mag, Real_t* alpha_mag)
{
    	int32_t p, q, r, idxr, idxq, idxp, i, j, k, l;
    	Real_t VMag[3], temp_mag[3][3][3][3], alpha_mag_temp, Delta0_Mag[3], QGGMRF_Params_Mag[3], gradmagavg[3], gradmagforw[3];
	Real_arr_t ****NeighAlphaMag;

	*alpha_mag = 0;

	compute_magcrossprodtran (ObjPtr->ErrorPotMag, grad_mag, ObjPtr->MagFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, -1.0);

	NeighAlphaMag = (Real_arr_t****)multialloc(sizeof(Real_arr_t), 4, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, 3);
  	memset(&(NeighAlphaMag[0][0][0][0]), 0, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*3*sizeof(Real_arr_t));

	gradmagavg[0] = 0; gradmagavg[1] = 0; gradmagavg[2] = 0; 	
	gradmagforw[0] = 0; gradmagforw[1] = 0; gradmagforw[2] = 0; 	
	for (i = 0; i < ObjPtr->N_z; i++)
	for (j = 0; j < ObjPtr->N_y; j++)
	for (k = 0; k < ObjPtr->N_x; k++)
	{
		gradmagforw[0] += fabs(grad_mag[i][j][k][0]);		
		gradmagforw[1] += fabs(grad_mag[i][j][k][1]);		
		gradmagforw[2] += fabs(grad_mag[i][j][k][2]);		
        	
		VMag[0] = ObjPtr->Magnetization[i][j][k][0]; /*Store the present value of the voxel*/
        	VMag[1] = ObjPtr->Magnetization[i][j][k][1]; /*Store the present value of the voxel*/
        	VMag[2] = ObjPtr->Magnetization[i][j][k][2]; /*Store the present value of the voxel*/

		grad_mag[i][j][k][0] *= InpPtr->ADMM_mu;		
		grad_mag[i][j][k][1] *= InpPtr->ADMM_mu;		
		grad_mag[i][j][k][2] *= InpPtr->ADMM_mu;		

	 	for (p = 0; p < 3; p++)
	 	{
#ifdef CIRC_BOUND_COND
		idxp = (i + p - 1 + ObjPtr->N_z) % ObjPtr->N_z;
#else
		idxp = i + p - 1;
		if (idxp >= 0 && idxp < ObjPtr->N_z){
#endif
	 		for (q = 0; q < 3; q++)
         		{
#ifdef CIRC_BOUND_COND
	 		idxq = (j + q - 1 + ObjPtr->N_y) % ObjPtr->N_y;
#else
	 		idxq = j + q - 1;
               		if(idxq >= 0 && idxq < ObjPtr->N_y){
#endif
				for (r = 0; r < 3; r++)
				{
#ifdef CIRC_BOUND_COND
		    		idxr = (k + r - 1 + ObjPtr->N_x) % ObjPtr->N_x;
#else
		    		idxr = k + r - 1;
                    		if(idxr >= 0 && idxr < ObjPtr->N_x){
#endif
      					Delta0_Mag[0] = (VMag[0] - ObjPtr->Magnetization[idxp][idxq][idxr][0]);
      					Delta0_Mag[1] = (VMag[1] - ObjPtr->Magnetization[idxp][idxq][idxr][1]);
      					Delta0_Mag[2] = (VMag[2] - ObjPtr->Magnetization[idxp][idxq][idxr][2]);

					for (l = 0; l < 3; l++)
					{
      						if(Delta0_Mag[l] != 0)	
      							QGGMRF_Params_Mag[l] = QGGMRF_Derivative(Delta0_Mag[l], InpPtr->Mag_Sigma_Q[l], InpPtr->Mag_Sigma_Q_P[l], ObjPtr->Mag_C[l])/(Delta0_Mag[l]);
	      					else 
        						QGGMRF_Params_Mag[l] = QGGMRF_SecondDerivative(InpPtr->Mag_Sigma_Q[l], ObjPtr->Mag_C[l]);
      					}
      
					temp_mag[p][q][r][0] = InpPtr->Spatial_Filter[p][q][r]*QGGMRF_Params_Mag[0];
					temp_mag[p][q][r][1] = InpPtr->Spatial_Filter[p][q][r]*QGGMRF_Params_Mag[1];
					temp_mag[p][q][r][2] = InpPtr->Spatial_Filter[p][q][r]*QGGMRF_Params_Mag[2];
     					
					grad_mag[i][j][k][0] += temp_mag[p][q][r][0]*Delta0_Mag[0];
      					grad_mag[i][j][k][1] += temp_mag[p][q][r][1]*Delta0_Mag[1];
      					grad_mag[i][j][k][2] += temp_mag[p][q][r][2]*Delta0_Mag[2];
      				
#ifndef CIRC_BOUND_COND
				}
#endif
               			}
#ifndef CIRC_BOUND_COND
			}
#endif
			}
#ifndef CIRC_BOUND_COND
		}
#endif
		}

	 	for (p = 0; p < 3; p++){
#ifdef CIRC_BOUND_COND
		idxp = (i + p - 1 + ObjPtr->N_z) % ObjPtr->N_z;
#else
		idxp = i + p - 1;
		if (idxp >= 0 && idxp < ObjPtr->N_z){
#endif
	 		for (q = 0; q < 3; q++){
#ifdef CIRC_BOUND_COND
	 		idxq = (j + q - 1 + ObjPtr->N_y) % ObjPtr->N_y;
#else
	 		idxq = j + q - 1;
               		if(idxq >= 0 && idxq < ObjPtr->N_y){
#endif
				for (r = 0; r < 3; r++){
#ifdef CIRC_BOUND_COND
		    		idxr = (k + r - 1 + ObjPtr->N_x) % ObjPtr->N_x;
#else
		    		idxr = k + r - 1;
                    		if(idxr >= 0 && idxr < ObjPtr->N_x){
#endif
					if (p != 1 || q != 1 || r != 1)
					{
						NeighAlphaMag[i][j][k][0] += grad_mag[i][j][k][0]*temp_mag[p][q][r][0];
						NeighAlphaMag[i][j][k][1] += grad_mag[i][j][k][1]*temp_mag[p][q][r][1];
						NeighAlphaMag[i][j][k][2] += grad_mag[i][j][k][2]*temp_mag[p][q][r][2];
						
						NeighAlphaMag[idxp][idxq][idxr][0] += -grad_mag[i][j][k][0]*temp_mag[p][q][r][0];
						NeighAlphaMag[idxp][idxq][idxr][1] += -grad_mag[i][j][k][1]*temp_mag[p][q][r][1];
						NeighAlphaMag[idxp][idxq][idxr][2] += -grad_mag[i][j][k][2]*temp_mag[p][q][r][2];
					}
#ifndef CIRC_BOUND_COND
				}
#endif
				}
#ifndef CIRC_BOUND_COND
			}
#endif
			}
#ifndef CIRC_BOUND_COND
		}
#endif
		}
						
		gradmagavg[0] += fabs(grad_mag[i][j][k][0]);		
		gradmagavg[1] += fabs(grad_mag[i][j][k][1]);		
		gradmagavg[2] += fabs(grad_mag[i][j][k][2]);		
	}

	gradmagavg[0] /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
	gradmagavg[1] /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
	gradmagavg[2] /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
	
	gradmagforw[0] /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
	gradmagforw[1] /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
	gradmagforw[2] /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);

	for (i = 0; i < ObjPtr->N_z; i++)
	for (j = 0; j < ObjPtr->N_y; j++)
	for (k = 0; k < ObjPtr->N_x; k++)
	{
		(*alpha_mag) += NeighAlphaMag[i][j][k][0]*grad_mag[i][j][k][0];	
		(*alpha_mag) += NeighAlphaMag[i][j][k][1]*grad_mag[i][j][k][1];	
		(*alpha_mag) += NeighAlphaMag[i][j][k][2]*grad_mag[i][j][k][2];	
	}
	
	compute_magcrossprodstep (grad_mag, ObjPtr->MagFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, &alpha_mag_temp);
	(*alpha_mag) += InpPtr->ADMM_mu*alpha_mag_temp;
	
	alpha_mag_temp = 0;
	for (i = 0; i < ObjPtr->N_z; i++)
	for (j = 0; j < ObjPtr->N_y; j++)
	for (k = 0; k < ObjPtr->N_x; k++)
	{
		alpha_mag_temp += grad_mag[i][j][k][0]*grad_mag[i][j][k][0];	
		alpha_mag_temp += grad_mag[i][j][k][1]*grad_mag[i][j][k][1];	
		alpha_mag_temp += grad_mag[i][j][k][2]*grad_mag[i][j][k][2];	
	}

	*alpha_mag = alpha_mag_temp/(*alpha_mag + EPSILON_ERROR);
	
	fprintf(InpPtr->debug_file_ptr, "Average of absolute value of gradient is (x, y, z) = (%e, %e, %e) and stepsize is (mag) = (%e)\n", gradmagavg[0], gradmagavg[1], gradmagavg[2], *alpha_mag);
	fprintf(InpPtr->debug_file_ptr, "Average of absolute value of forward part of gradient is (x, y, z) = (%e, %e, %e)\n", gradmagforw[0], gradmagforw[1], gradmagforw[2]);

	multifree(NeighAlphaMag, 4);
}

void compute_elecprodtran (Real_arr_t*** elecobjinp, Real_arr_t*** elecobjout, Real_arr_t**** elecimp, FFTStruct *fftptr, int32_t N_z, int32_t N_y, int32_t N_x, Real_t tran_sign)
{
	int32_t i, j, k;
	int64_t idx, idx_i, idx_j, idx_k;
	Real_t N, real_rho, imag_rho;
	N = fftptr->z_num*fftptr->y_num*fftptr->x_num;

	fftw_complex* fftforw_elecarr = fftptr->fftforw_elecarr;
	fftw_plan* fftforw_elecplan = &(fftptr->fftforw_elecplan);
	fftw_complex* fftback_elecarr = fftptr->fftback_elecarr;
	fftw_plan* fftback_elecplan = &(fftptr->fftback_elecplan);

	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;

		if (i >= fftptr->z0 && i < fftptr->z0 + N_z && j >= fftptr->y0 && j < fftptr->y0 + N_y && k >= fftptr->x0 && k < fftptr->x0 + N_x)
		{
			idx_i = i - fftptr->z0;
			idx_j = j - fftptr->y0;
			idx_k = k - fftptr->x0;
			
			fftforw_elecarr[idx][0] = elecobjinp[idx_i][idx_j][idx_k];
			fftforw_elecarr[idx][1] = 0;
		}
		else
		{
			fftforw_elecarr[idx][0] = 0;
			fftforw_elecarr[idx][1] = 0;
		}
	}
	
	fftw_execute(fftforw_elecplan[0]);

	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;

		cmplx_mult (&real_rho, &imag_rho, elecimp[i][j][k][0], tran_sign*elecimp[i][j][k][1], fftforw_elecarr[idx][0], fftforw_elecarr[idx][1]);

		fftback_elecarr[idx][0] = (real_rho);
		fftback_elecarr[idx][1] = (imag_rho);
	}
	
	fftw_execute(fftback_elecplan[0]);
	
	for (i = 0; i < N_z; i++) for (j = 0; j < N_y; j++) for (k = 0; k < N_x; k++)
	{
		idx = (i + fftptr->z0)*fftptr->y_num*fftptr->x_num + (j + fftptr->y0)*fftptr->x_num + (k + fftptr->x0);
		elecobjout[i][j][k] = fftback_elecarr[idx][0]/N;
	}
	 
}

void compute_elecprodstep (Real_arr_t*** elecobjinp, Real_arr_t**** elecimp, FFTStruct *fftptr, int32_t N_z, int32_t N_y, int32_t N_x, Real_t* alpha_elec)
{
	int32_t i, j, k;
	int64_t idx, idx_i, idx_j, idx_k;
	Real_t real_1, imag_1, real_rho, imag_rho, N;
	
	fftw_complex* fftforw_elecarr = fftptr->fftforw_elecarr;
	fftw_plan* fftforw_elecplan = &(fftptr->fftforw_elecplan);
	fftw_complex* fftback_elecarr = fftptr->fftback_elecarr;
	fftw_plan* fftback_elecplan = &(fftptr->fftback_elecplan);
	
	N = fftptr->z_num*fftptr->y_num*fftptr->x_num;
	
	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;

		if (i >= fftptr->z0 && i < fftptr->z0 + N_z && j >= fftptr->y0 && j < fftptr->y0 + N_y && k >= fftptr->x0 && k < fftptr->x0 + N_x)
		{	
			idx_i = i - fftptr->z0;
			idx_j = j - fftptr->y0;
			idx_k = k - fftptr->x0;
			
			fftforw_elecarr[idx][0] = elecobjinp[idx_i][idx_j][idx_k];
			fftforw_elecarr[idx][1] = 0;
		}
		else
		{
			fftforw_elecarr[idx][0] = 0;
			fftforw_elecarr[idx][1] = 0;
		}
	}
	
	fftw_execute(fftforw_elecplan[0]);

	for (i = 0; i < fftptr->z_num; i++) for (j = 0; j < fftptr->y_num; j++) for (k = 0; k < fftptr->x_num; k++)
	{
		idx = i*fftptr->y_num*fftptr->x_num + j*fftptr->x_num + k;
		
		cmplx_mult (&real_1, &imag_1, elecimp[i][j][k][0], -elecimp[i][j][k][1], elecimp[i][j][k][0], elecimp[i][j][k][1]);
		cmplx_mult (&real_rho, &imag_rho, real_1, imag_1, fftforw_elecarr[idx][0], fftforw_elecarr[idx][1]);
		
		fftback_elecarr[idx][0] = real_rho;
		fftback_elecarr[idx][1] = imag_rho;
	}

	fftw_execute(fftback_elecplan[0]);

	*alpha_elec = 0;
	for (i = 0; i < N_z; i++) for (j = 0; j < N_y; j++) for (k = 0; k < N_x; k++)
	{
		idx = (i + fftptr->z0)*fftptr->y_num*fftptr->x_num + (j + fftptr->y0)*fftptr->x_num + (k + fftptr->x0);
		
		*alpha_elec += elecobjinp[i][j][k]*fftback_elecarr[idx][0];	
	}
	*alpha_elec /= N;

	
}	


void compute_elec_gradstep(ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, Real_arr_t*** grad_elec, Real_t* alpha_elec)
{
    	int32_t p, q, r, idxr, idxq, idxp, i, j, k;
    	Real_t VElec, temp_elec[3][3][3], alpha_elec_temp, Delta0_Elec, QGGMRF_Params_Elec, gradelecavg, gradelecforw;
	Real_arr_t ***NeighAlphaElec;

	*alpha_elec = 0;

	compute_elecprodtran (ObjPtr->ErrorPotElec, grad_elec, ObjPtr->ElecFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, -1.0);

	NeighAlphaElec = (Real_arr_t***)multialloc(sizeof(Real_arr_t), 3, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x);
  	memset(&(NeighAlphaElec[0][0][0]), 0, ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x*sizeof(Real_arr_t));

	gradelecavg = 0;	
	gradelecforw = 0;	
	for (i = 0; i < ObjPtr->N_z; i++)
	for (j = 0; j < ObjPtr->N_y; j++)
	for (k = 0; k < ObjPtr->N_x; k++)
	{
		gradelecforw += fabs(grad_elec[i][j][k]);		
        	VElec = ObjPtr->ChargeDensity[i][j][k]; /*Store the present value of the voxel*/

		grad_elec[i][j][k] *= -InpPtr->ADMM_mu;		

	 	for (p = 0; p < 3; p++)
	 	{
		idxp = i + p - 1;
		if (idxp >= 0 && idxp < ObjPtr->N_z)
		{
	 		for (q = 0; q < 3; q++)
         		{
	 		idxq = j + q - 1;
               		if(idxq >= 0 && idxq < ObjPtr->N_y)
         		{
				for (r = 0; r < 3; r++)
				{
		    		idxr = k + r - 1;
                    		if(idxr >= 0 && idxr < ObjPtr->N_x){
      					Delta0_Elec = (VElec - ObjPtr->ChargeDensity[idxp][idxq][idxr]);

      					if(Delta0_Elec != 0)
      						QGGMRF_Params_Elec = QGGMRF_Derivative(Delta0_Elec, InpPtr->Elec_Sigma_Q, InpPtr->Elec_Sigma_Q_P, ObjPtr->Elec_C)/(Delta0_Elec);
      					else 
        					QGGMRF_Params_Elec = QGGMRF_SecondDerivative(InpPtr->Elec_Sigma_Q, ObjPtr->Elec_C);

      					temp_elec[p][q][r] = InpPtr->Spatial_Filter[p][q][r]*QGGMRF_Params_Elec;
      					grad_elec[i][j][k] += temp_elec[p][q][r]*Delta0_Elec;
      				
				}
               			}
			}
			}
		}
		}

	 	for (p = 0; p < 3; p++){
		idxp = i + p - 1;
		if (idxp >= 0 && idxp < ObjPtr->N_z){
	 		for (q = 0; q < 3; q++){
	 		idxq = j + q - 1;
               		if(idxq >= 0 && idxq < ObjPtr->N_y){
				for (r = 0; r < 3; r++){
		    		idxr = k + r - 1;
                    		if(idxr >= 0 && idxr < ObjPtr->N_x){
					if (p != 1 || q != 1 || r != 1)
					{
						NeighAlphaElec[i][j][k] += grad_elec[i][j][k]*temp_elec[p][q][r];
						NeighAlphaElec[idxp][idxq][idxr] += -grad_elec[i][j][k]*temp_elec[p][q][r];
					}
				}
			}
			}
		}
		}
		}
						
		gradelecavg += fabs(grad_elec[i][j][k]);		
	}

	gradelecavg /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);
	gradelecforw /= (ObjPtr->N_z*ObjPtr->N_y*ObjPtr->N_x);

		
	for (i = 0; i < ObjPtr->N_z; i++)
	for (j = 0; j < ObjPtr->N_y; j++)
	for (k = 0; k < ObjPtr->N_x; k++)
	{
		(*alpha_elec) += NeighAlphaElec[i][j][k]*grad_elec[i][j][k];
	}
	
	compute_elecprodstep (grad_elec, ObjPtr->ElecFilt, fftptr, ObjPtr->N_z, ObjPtr->N_y, ObjPtr->N_x, &alpha_elec_temp);
	(*alpha_elec) += InpPtr->ADMM_mu*alpha_elec_temp;

	alpha_elec_temp = 0;
	for (i = 0; i < ObjPtr->N_z; i++)
	for (j = 0; j < ObjPtr->N_y; j++)
	for (k = 0; k < ObjPtr->N_x; k++)
	{
		alpha_elec_temp += grad_elec[i][j][k]*grad_elec[i][j][k];	
	}

	*alpha_elec = alpha_elec_temp/(*alpha_elec + EPSILON_ERROR);
	
	fprintf(InpPtr->debug_file_ptr, "Average of absolute value of gradient is (elec) = (%e) and stepsize is (elec) = (%e)\n", gradelecavg, *alpha_elec);
	fprintf(InpPtr->debug_file_ptr, "Average of absolute value of forward part of gradient is (elec) = (%e)\n", gradelecforw);

	multifree(NeighAlphaElec, 3);
}


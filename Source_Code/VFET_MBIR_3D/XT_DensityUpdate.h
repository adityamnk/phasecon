#ifndef XT_DENSITYUPDATE_H


void compute_gradient_stepsize(ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, Real_arr_t**** grad_mag, Real_arr_t*** grad_elec, Real_t* alpha_mag, Real_t* alpha_elec);
void compute_crossprodtran (Real_arr_t**** magobjinp, Real_arr_t*** elecobjinp, Real_arr_t**** magobjout, Real_arr_t*** elecobjout, Real_arr_t**** magimp, Real_arr_t**** elecimp, FFTStruct *fftptr, int32_t N_z, int32_t N_y, int32_t N_x, Real_t tran_sign);

#define XT_DENSITYUPDATE_H
#endif

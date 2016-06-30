#ifndef XT_DENSITYUPDATE_H

void compute_magcrossprodtran (Real_arr_t**** magobjinp, Real_arr_t**** magobjout, Real_arr_t**** magimp, FFTStruct *fftptr, int32_t N_z, int32_t N_y, int32_t N_x, Real_t tran_sign);
void compute_mag_gradstep(ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, Real_arr_t**** grad_mag, Real_t* alpha_mag);
void compute_elecprodtran (Real_arr_t*** elecobjinp, Real_arr_t*** elecobjout, Real_arr_t**** elecimp, FFTStruct *fftptr, int32_t N_z, int32_t N_y, int32_t N_x, Real_t tran_sign);
void compute_elec_gradstep(ScannedObject* ObjPtr, TomoInputs* InpPtr, FFTStruct* fftptr, Real_arr_t*** grad_elec, Real_t* alpha_elec);

#define XT_DENSITYUPDATE_H
#endif

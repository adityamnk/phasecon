#ifndef XT_CMPLXPROJEST_H
#define XT_CMPLXPROJEST_H

void estimate_complex_projection (Real_arr_t** measurements_real, Real_arr_t** measurements_imag, Real_arr_t** omega_real, Real_arr_t** omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t*** z_real, Real_arr_t*** z_imag, Real_arr_t** Lambda, Real_arr_t** proj_real, Real_arr_t** proj_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** v_real, Real_arr_t** v_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, Real_t NMS_rho, Real_t NMS_chi, Real_t NMS_gamma, Real_t NMS_sigma, Real_t NMS_thresh, int32_t NMS_iters, Real_t steepdes_thresh, int32_t steepdes_iters, Real_t pret_thresh, int32_t pret_iters, Real_t mu, Real_t nu, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan);

#endif

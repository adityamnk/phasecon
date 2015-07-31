#ifndef XT_PHASERET_H
#define XT_PHASERET_H

Real_t steepest_descent_iter (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** Lambda, Real_arr_t** v_real, Real_arr_t** v_imag, Real_t nu, int32_t rows, int32_t cols, fftw_complex *fftforward_arr, fftw_plan *fftforward_plan, fftw_complex *fftbackward_arr, fftw_plan *fftbackward_plan);
Real_t compute_cost (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** Lambda, Real_arr_t** v_real, Real_arr_t** v_imag, Real_t nu, int32_t rows, int32_t cols, fftw_complex *fftarr, fftw_plan *p);

#endif

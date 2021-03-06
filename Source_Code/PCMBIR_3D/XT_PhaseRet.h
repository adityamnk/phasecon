#ifndef XT_PHASERET_H
#define XT_PHASERET_H

void compute_phase_projection (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan, Real_t light_wavelength, Real_t obj2det_distance, Real_arr_t** FresnelFreqWin); 
Real_t steepest_descent_iter (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** Lambda, Real_arr_t** v_real, Real_arr_t** v_imag, Real_t nu, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex *fftforward_arr, fftw_plan *fftforward_plan, fftw_complex *fftbackward_arr, fftw_plan *fftbackward_plan, Real_t light_wavelength, Real_t obj2det_distance, Real_arr_t** FresnelFreqWin);
Real_t compute_cost (Real_arr_t** y_real, Real_arr_t** y_imag, Real_arr_t** Omega_real, Real_arr_t** Omega_imag, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** w_real, Real_arr_t** w_imag, Real_arr_t** Lambda, Real_arr_t** v_real, Real_arr_t** v_imag, Real_t nu, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex *fftforw_arr, fftw_plan *fftforw_plan, fftw_complex *fftback_arr, fftw_plan *fftback_plan, Real_t light_wavelength, Real_t obj2det_distance, Real_arr_t** FresnelFreqWin);


#endif

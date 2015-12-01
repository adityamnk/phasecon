#ifndef XT_PAGANIN_H
#define XT_PAGANIN_H

void paganins_1mat_phase_retrieval (Real_arr_t** measurements, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t** z_real, Real_arr_t** z_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan, Real_t light_wavenumber, Real_t light_wavelength, Real_t obj2det_dist, Real_t pag_regparam);
void paganins_2mat_phase_retrieval (Real_arr_t** measurements, Real_arr_t** D_real, Real_arr_t** D_imag, Real_arr_t**  projlength, Real_arr_t** z_real, Real_arr_t** z_imag, int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan* fftforw_plan, fftw_complex* fftback_arr, fftw_plan* fftback_plan, Real_t light_wavenumber, Real_t light_wavelength, Real_t obj2det_dist, Real_t pag_regparam);

#endif

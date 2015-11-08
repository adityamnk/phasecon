#ifndef XT_FRESNELTRAN_H

#define XT_FRESNELTRAN_H
void compute_FresnelTran (int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan, Real_t light_wavelength, Real_t obj2det_distance);
void compute_HermFresnelTran (int32_t rows, int32_t cols, Real_t delta_rows, Real_t delta_cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan, Real_t light_wavelength, Real_t obj2det_distance);


#endif

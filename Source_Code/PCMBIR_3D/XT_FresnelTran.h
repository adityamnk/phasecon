#ifndef XT_FRESNELTRAN_H

#define XT_FRESNELTRAN_H
void compute_FresnelTran (int32_t rows, int32_t cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan);
void compute_HermFresnelTran (int32_t rows, int32_t cols, fftw_complex* fftforw_arr, fftw_plan *fftforw_plan, fftw_complex* fftback_arr, fftw_plan *fftback_plan);


#endif

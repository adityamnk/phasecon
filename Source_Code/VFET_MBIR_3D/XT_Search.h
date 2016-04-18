#ifndef XT_SEARCH_H
#define XT_SEARCH_H

int32_t Nelder_Mead_Simplex_2DSearch (Real_arr_t* zr, Real_arr_t* zi, Real_t ar, Real_t ai, Real_t br, Real_t bi, Real_t NMS_rho, Real_t NMS_chi, Real_t NMS_gamma, Real_t NMS_sigma, Real_t NMS_threshold, int32_t NMS_MaxIter, Real_t mu, Real_t nu, FILE* debug_file_ptr);
Real_t Cost_NMS (Real_t zr, Real_t zi, Real_t ar, Real_t ai, Real_t br, Real_t bi, Real_t mu, Real_t nu);

#endif


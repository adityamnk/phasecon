
Real_t updateVoxels (int32_t slice_begin, int32_t slice_end, int32_t xy_begin, int32_t xy_end, int32_t* x_rand_select, int32_t* y_rand_select, Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_arr_t*** ErrorSino_Unflip_x, Real_arr_t*** ErrorSino_Flip_x, Real_arr_t*** ErrorSino_Unflip_y, Real_arr_t*** ErrorSino_Flip_y, Real_arr_t** DetectorResponse_XY, int32_t Iter, long int *zero_count, Real_arr_t** MagUpdateMap, uint8_t** Mask);

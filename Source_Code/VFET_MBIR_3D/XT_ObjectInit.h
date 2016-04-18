#ifndef XT_OBJECTINIT_H
#define XT_OBJECTINIT_H


int32_t initObject (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
int init_minmax_object (ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
void gen_data_GroundTruth (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);

#endif

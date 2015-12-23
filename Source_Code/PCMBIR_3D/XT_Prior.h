#ifndef XT_PRIOR_H
#define XT_PRIOR_H

Real_t Mag_QGGMRF_Spatial_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Phase_QGGMRF_Spatial_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Mag_QGGMRF_Temporal_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Phase_QGGMRF_Temporal_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Mag_QGGMRF_Spatial_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Phase_QGGMRF_Spatial_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Mag_QGGMRF_Temporal_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Phase_QGGMRF_Temporal_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Mag_QGGMRF_Spatial_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Phase_QGGMRF_Spatial_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Mag_QGGMRF_Temporal_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Phase_QGGMRF_Temporal_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr);
Real_t Mag_FunctionalSubstitution(Real_t V, Real_t THETA1, Real_t THETA2, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Spatial_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood[NHOOD_TIME_MAXDIM-1], bool Spatial_BDFlag[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], bool Time_BDFlag[NHOOD_TIME_MAXDIM-1]);
Real_t Phase_FunctionalSubstitution(Real_t V, Real_t THETA1, Real_t THETA2, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Spatial_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood[NHOOD_TIME_MAXDIM-1], bool Spatial_BDFlag[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], bool Time_BDFlag[NHOOD_TIME_MAXDIM-1]);
void FunctionalSubstitution(Real_t* V_delta, Real_t* V_beta, Real_t THETA1_delta, Real_t THETA1_beta, Real_t THETA2_delta, Real_t THETA2_beta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Spatial_Nhood_delta[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood_delta[NHOOD_TIME_MAXDIM-1], Real_t Spatial_Nhood_beta[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood_beta[NHOOD_TIME_MAXDIM-1], bool Spatial_BDFlag[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], bool Time_BDFlag[NHOOD_TIME_MAXDIM-1], Real_t DecorrTran[2][2]);


#endif

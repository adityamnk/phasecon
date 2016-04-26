#ifndef XT_PRIOR_H
#define XT_PRIOR_H


Real_t QGGMRF_Value(Real_t delta, Real_t Sigma_Q, Real_t Sigma_Q_P, Real_t C);
Real_t QGGMRF_Derivative(Real_t delta, Real_t Sigma_Q, Real_t Sigma_Q_P, Real_t C);
Real_t QGGMRF_SecondDerivative(Real_t Sigma_Q, Real_t C);
void FunctionalSubstitution(Real_t *VMag, Real_t *VElec, Real_t THETA1_Mag[3], Real_t THETA2_Mag[3][3], Real_t THETA1_Elec, Real_t THETA2_Elec, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Mag_Nhood[3][3][3][3], Real_t Elec_Nhood[3][3][3], bool BDFlag[3][3][3]);

#endif

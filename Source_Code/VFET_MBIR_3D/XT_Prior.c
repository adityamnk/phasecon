#include <math.h>
#include <stdio.h>
#include "XT_Constants.h"
#include "XT_Structures.h"

/*Computes the qGGMRF spatial prior cost value at delta = x_i - x_j. i & j being the voxel and its neighbor*/
Real_t QGGMRF_Value(Real_t delta, Real_t Sigma_Q, Real_t Sigma_Q_P, Real_t C)
{
  return ((pow(fabs(delta),MRF_Q)/Sigma_Q)/(C + pow(fabs(delta),MRF_Q - MRF_P)/Sigma_Q_P));
}

/*Computes the qGGMRF spatial prior derivative at delta = x_i - x_j. i & j being the voxel and its neighbor*/
Real_t QGGMRF_Derivative(Real_t delta, Real_t Sigma_Q, Real_t Sigma_Q_P, Real_t C)
{
  Real_t temp1,temp2,temp3;
  temp1=pow(fabs(delta),MRF_Q - MRF_P)/(Sigma_Q_P);
  temp2=pow(fabs(delta),MRF_Q - 1);
  temp3 = C + temp1;
  if(delta < 0)
  return ((-1*temp2/(temp3*Sigma_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else
  {
    return ((temp2/(temp3*Sigma_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  }
}

/*Computes the qGGMRF spatial prior second derivative at delta = 0*/
Real_t QGGMRF_SecondDerivative(Real_t Sigma_Q, Real_t C)
{
  return MRF_Q/(Sigma_Q*C);
}


void matinv (Real_t A[3][3], Real_t result[3][3])
{
	Real_t determinant = A[0][0]*(A[1][1]*A[2][2]-A[2][1]*A[1][2])
                        -A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                        +A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
	Real_t invdet = 1.0/determinant;
	result[0][0] =  (A[1][1]*A[2][2]-A[2][1]*A[1][2])*invdet;
	result[0][1] = -(A[0][1]*A[2][2]-A[0][2]*A[2][1])*invdet;
	result[0][2] =  (A[0][1]*A[1][2]-A[0][2]*A[1][1])*invdet;
	result[1][0] = -(A[1][0]*A[2][2]-A[1][2]*A[2][0])*invdet;
	result[1][1] =  (A[0][0]*A[2][2]-A[0][2]*A[2][0])*invdet;
	result[1][2] = -(A[0][0]*A[1][2]-A[1][0]*A[0][2])*invdet;
	result[2][0] =  (A[1][0]*A[2][1]-A[2][0]*A[1][1])*invdet;
	result[2][1] = -(A[0][0]*A[2][1]-A[2][0]*A[0][1])*invdet;
	result[2][2] =  (A[0][0]*A[1][1]-A[1][0]*A[0][1])*invdet;
}


void MagFunctionalSubstitutionConstPrior(Real_t *VMag, Real_t THETA1_Mag[3], Real_t THETA2_Mag[3][3], ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t MagPrior[3])
{
  Real_t u_mag[3], temp1_mag[3], temp2_mag[3][3];
  int32_t i;

  temp1_mag[0] = 0; temp1_mag[1] = 0; temp1_mag[2] = 0; 
  temp2_mag[0][0] = 0; temp2_mag[0][1] = 0; temp2_mag[0][2] = 0; 
  temp2_mag[1][0] = 0; temp2_mag[1][1] = 0; temp2_mag[1][2] = 0;
  temp2_mag[2][0] = 0; temp2_mag[2][1] = 0; temp2_mag[2][2] = 0;

  /*Need to Loop this for multiple iterations of substitute function*/
  temp1_mag[0] += TomoInputsPtr->ADMM_mu*MagPrior[0];
  temp1_mag[1] += TomoInputsPtr->ADMM_mu*MagPrior[1];
  temp1_mag[2] += TomoInputsPtr->ADMM_mu*MagPrior[2];
      	
  temp1_mag[0] = (temp1_mag[0] + (THETA2_Mag[0][0]*VMag[0] + THETA2_Mag[0][1]*VMag[1] + THETA2_Mag[0][2]*VMag[2]) - THETA1_Mag[0]);
  temp1_mag[1] = (temp1_mag[1] + (THETA2_Mag[1][0]*VMag[0] + THETA2_Mag[1][1]*VMag[1] + THETA2_Mag[1][2]*VMag[2]) - THETA1_Mag[1]);
  temp1_mag[2] = (temp1_mag[2] + (THETA2_Mag[2][0]*VMag[0] + THETA2_Mag[2][1]*VMag[1] + THETA2_Mag[2][2]*VMag[2]) - THETA1_Mag[2]);

  for (i = 0; i < 3; i++)
	THETA2_Mag[i][i] += TomoInputsPtr->ADMM_mu;

  matinv(THETA2_Mag, temp2_mag);

  u_mag[0] = temp1_mag[0]*temp2_mag[0][0] + temp1_mag[1]*temp2_mag[0][1] + temp1_mag[2]*temp2_mag[0][2];
  u_mag[1] = temp1_mag[0]*temp2_mag[1][0] + temp1_mag[1]*temp2_mag[1][1] + temp1_mag[2]*temp2_mag[1][2];
  u_mag[2] = temp1_mag[0]*temp2_mag[2][0] + temp1_mag[1]*temp2_mag[2][1] + temp1_mag[2]*temp2_mag[2][2];

  VMag[0] = VMag[0] + TomoInputsPtr->alpha*(u_mag[0] - VMag[0]);
  VMag[1] = VMag[1] + TomoInputsPtr->alpha*(u_mag[1] - VMag[1]);
  VMag[2] = VMag[2] + TomoInputsPtr->alpha*(u_mag[2] - VMag[2]);

}


void ElecFunctionalSubstitutionConstPrior(Real_t *VElec, Real_t THETA1_Elec, Real_t THETA2_Elec, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t ElecPrior)
{
  Real_t u_elec, temp1_elec, temp2_elec;

  temp1_elec = 0;
  temp2_elec = 0;
  temp1_elec += TomoInputsPtr->ADMM_mu*ElecPrior;
  temp2_elec += TomoInputsPtr->ADMM_mu;

  u_elec = (temp1_elec + (THETA2_Elec*VElec[0]) - THETA1_Elec)/(temp2_elec + THETA2_Elec);
  
  VElec[0] = VElec[0] + TomoInputsPtr->alpha*(u_elec - VElec[0]);
  if (VElec[0] <= 0)
  VElec[0] = 0;
}



































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

/*Computes the voxel update and returns it. V is the present value of voxel.
THETA1 and THETA2 are the values used in voxel update. Spatial_Nhood and Time_Nhood gives the
values of voxels in the neighborhood of V. Time_BDFlag and Spatial_BDFlag are masks which determine
whether a neighbor should be included in the neighorhood or not.*/
void FunctionalSubstitution(Real_t *VMag, Real_t *VElec, Real_t THETA1_Mag[3], Real_t THETA2_Mag[3][3], Real_t THETA1_Elec, Real_t THETA2_Elec, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Mag_Nhood[3][3][3][3], Real_t Elec_Nhood[3][3][3], bool BDFlag[3][3][3])
{
  Real_t u_mag[3], u_elec, temp1_mag[3], temp1_elec, temp2_mag[3][3], temp2_elec, Delta0_Mag[3], Delta0_Elec;
  Real_t QGGMRF_Params_Mag[3], QGGMRF_Params_Elec;
  int32_t i,j,k,l;

  temp1_mag[0] = 0; temp1_mag[1] = 0; temp1_mag[2] = 0; temp1_elec = 0;
  temp2_mag[0][0] = 0; temp2_mag[0][1] = 0; temp2_mag[0][2] = 0; 
  temp2_mag[1][0] = 0; temp2_mag[1][1] = 0; temp2_mag[1][2] = 0;
  temp2_mag[2][0] = 0; temp2_mag[2][1] = 0; temp2_mag[2][2] = 0;
  temp2_elec = 0;

  /*Need to Loop this for multiple iterations of substitute function*/
  for (i=0; i < 3; i++)
  for (j=0; j < 3; j++)
  for (k=0; k < 3; k++)
  {
    if(BDFlag[i][j][k] == true && (i != 1 || j != 1 || k != 1))
    {
      	Delta0_Mag[0] = (VMag[0] - Mag_Nhood[i][j][k][0]);
      	Delta0_Mag[1] = (VMag[1] - Mag_Nhood[i][j][k][1]);
      	Delta0_Mag[2] = (VMag[2] - Mag_Nhood[i][j][k][2]);
      	Delta0_Elec = (VElec[0] - Elec_Nhood[i][j][k]);

	for (l = 0; l < 3; l++)
	{
      		if(Delta0_Mag[l] != 0)
      			QGGMRF_Params_Mag[l] = QGGMRF_Derivative(Delta0_Mag[l], TomoInputsPtr->Mag_Sigma_Q[l], TomoInputsPtr->Mag_Sigma_Q_P[l], ScannedObjectPtr->Mag_C[l])/(Delta0_Mag[l]);
	      	else 
        		QGGMRF_Params_Mag[l] = QGGMRF_SecondDerivative(TomoInputsPtr->Mag_Sigma_Q[l], ScannedObjectPtr->Mag_C[l]);
      	}
      
      	if(Delta0_Elec != 0)
      		QGGMRF_Params_Elec = QGGMRF_Derivative(Delta0_Elec, TomoInputsPtr->Elec_Sigma_Q, TomoInputsPtr->Elec_Sigma_Q_P, ScannedObjectPtr->Elec_C)/(Delta0_Elec);
      	else 
        	QGGMRF_Params_Elec = QGGMRF_SecondDerivative(TomoInputsPtr->Elec_Sigma_Q, ScannedObjectPtr->Elec_C);

      	temp1_mag[0] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Mag[0]*Mag_Nhood[i][j][k][0];
      	temp1_mag[1] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Mag[1]*Mag_Nhood[i][j][k][1];
      	temp1_mag[2] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Mag[2]*Mag_Nhood[i][j][k][2];
      	temp1_elec += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Elec*Elec_Nhood[i][j][k];
      	
      	temp2_mag[0][0] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Mag[0];
	temp2_mag[1][1] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Mag[1];
	temp2_mag[2][2] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Mag[2];
      	temp2_elec += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params_Elec;
     }
  }

  temp1_mag[0] = (temp1_mag[0] + (THETA2_Mag[0][0]*VMag[0] + THETA2_Mag[0][1]*VMag[1] + THETA2_Mag[0][2]*VMag[2]) - THETA1_Mag[0]);
  temp1_mag[1] = (temp1_mag[1] + (THETA2_Mag[1][0]*VMag[0] + THETA2_Mag[1][1]*VMag[1] + THETA2_Mag[1][2]*VMag[2]) - THETA1_Mag[1]);
  temp1_mag[2] = (temp1_mag[2] + (THETA2_Mag[2][0]*VMag[0] + THETA2_Mag[2][1]*VMag[1] + THETA2_Mag[2][2]*VMag[2]) - THETA1_Mag[2]);

  for (i = 0; i < 3; i++)
  for (j = 0; j < 3; j++)
	THETA2_Mag[i][j] += temp2_mag[i][j];

  matinv(THETA2_Mag, temp2_mag);

  u_mag[0] = temp1_mag[0]*temp2_mag[0][0] + temp1_mag[1]*temp2_mag[0][1] + temp1_mag[2]*temp2_mag[0][2];
  u_mag[1] = temp1_mag[0]*temp2_mag[1][0] + temp1_mag[1]*temp2_mag[1][1] + temp1_mag[2]*temp2_mag[1][2];
  u_mag[2] = temp1_mag[0]*temp2_mag[2][0] + temp1_mag[1]*temp2_mag[2][1] + temp1_mag[2]*temp2_mag[2][2];

  u_elec = (temp1_elec + (THETA2_Elec*VElec[0]) - THETA1_Elec)/(temp2_elec + THETA2_Elec);
  
  VMag[0] = VMag[0] + TomoInputsPtr->alpha*(u_mag[0] - VMag[0]);
  VMag[1] = VMag[1] + TomoInputsPtr->alpha*(u_mag[1] - VMag[1]);
  VMag[2] = VMag[2] + TomoInputsPtr->alpha*(u_mag[2] - VMag[2]);
  VElec[0] = VElec[0] + TomoInputsPtr->alpha*(u_elec - VElec[0]);

  #ifdef POSITIVITY_CONSTRAINT
  if (VMag[0] <= 0)
  VMag[0] = 0;
  if (VMag[1] <= 0)
  VMag[1] = 0;
  if (VMag[2] <= 0)
  VMag[2] = 0;
  if (VElec[0] <= 0)
  VElec[0] = 0;
  #endif
}











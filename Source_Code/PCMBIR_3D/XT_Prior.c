#include <math.h>
#include <stdio.h>
#include "XT_Constants.h"
#include "XT_Structures.h"

/*Computes the qGGMRF spatial prior cost value at delta = x_i - x_j. i & j being the voxel and its neighbor*/
Real_t Mag_QGGMRF_Spatial_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return ((pow(fabs(delta),MRF_Q)/TomoInputsPtr->Mag_Sigma_S_Q)/(ScannedObjectPtr->Mag_C_S + pow(fabs(delta),MRF_Q - MRF_P)/TomoInputsPtr->Mag_Sigma_S_Q_P));
}

Real_t Phase_QGGMRF_Spatial_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return ((pow(fabs(delta),MRF_Q)/TomoInputsPtr->Phase_Sigma_S_Q)/(ScannedObjectPtr->Phase_C_S + pow(fabs(delta),MRF_Q - MRF_P)/TomoInputsPtr->Phase_Sigma_S_Q_P));
}


/*Computes the qGGMRF temporal prior cost value at delta = x_i - x_j. i & j being the voxel and its neighbor*/
Real_t Mag_QGGMRF_Temporal_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return ((pow(fabs(delta),MRF_Q)/TomoInputsPtr->Mag_Sigma_T_Q)/(ScannedObjectPtr->Mag_C_T + pow(fabs(delta),MRF_Q - MRF_P)/TomoInputsPtr->Mag_Sigma_T_Q_P));
}

Real_t Phase_QGGMRF_Temporal_Value(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return ((pow(fabs(delta),MRF_Q)/TomoInputsPtr->Phase_Sigma_T_Q)/(ScannedObjectPtr->Phase_C_T + pow(fabs(delta),MRF_Q - MRF_P)/TomoInputsPtr->Phase_Sigma_T_Q_P));
}

/*Computes the qGGMRF spatial prior derivative at delta = x_i - x_j. i & j being the voxel and its neighbor*/
Real_t Mag_QGGMRF_Spatial_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  Real_t temp1,temp2,temp3;
  temp1=pow(fabs(delta),MRF_Q - MRF_P)/(TomoInputsPtr->Mag_Sigma_S_Q_P);
  temp2=pow(fabs(delta),MRF_Q - 1);
  temp3 = ScannedObjectPtr->Mag_C_S + temp1;
  if(delta < 0)
  return ((-1*temp2/(temp3*TomoInputsPtr->Mag_Sigma_S_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else
  {
    return ((temp2/(temp3*TomoInputsPtr->Mag_Sigma_S_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  }
}

Real_t Phase_QGGMRF_Spatial_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  Real_t temp1,temp2,temp3;
  temp1=pow(fabs(delta),MRF_Q - MRF_P)/(TomoInputsPtr->Phase_Sigma_S_Q_P);
  temp2=pow(fabs(delta),MRF_Q - 1);
  temp3 = ScannedObjectPtr->Phase_C_S + temp1;
  if(delta < 0)
  return ((-1*temp2/(temp3*TomoInputsPtr->Phase_Sigma_S_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else
  {
    return ((temp2/(temp3*TomoInputsPtr->Phase_Sigma_S_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  }
}

/*Computes the qGGMRF temporal prior derivative at delta = x_i - x_j. i & j being the voxel and its neighbor*/
Real_t Mag_QGGMRF_Temporal_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  Real_t temp1,temp2,temp3;
  temp1 = pow(fabs(delta),MRF_Q - MRF_P)/(TomoInputsPtr->Mag_Sigma_T_Q_P);
  temp2 = pow(fabs(delta),MRF_Q - 1);
  temp3 = ScannedObjectPtr->Mag_C_T + temp1;
  if(delta < 0)
  return ((-1*temp2/(temp3*TomoInputsPtr->Mag_Sigma_T_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else
  {
    return ((temp2/(temp3*TomoInputsPtr->Mag_Sigma_T_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  }
}

Real_t Phase_QGGMRF_Temporal_Derivative(Real_t delta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  Real_t temp1,temp2,temp3;
  temp1 = pow(fabs(delta),MRF_Q - MRF_P)/(TomoInputsPtr->Phase_Sigma_T_Q_P);
  temp2 = pow(fabs(delta),MRF_Q - 1);
  temp3 = ScannedObjectPtr->Phase_C_T + temp1;
  if(delta < 0)
  return ((-1*temp2/(temp3*TomoInputsPtr->Phase_Sigma_T_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  else
  {
    return ((temp2/(temp3*TomoInputsPtr->Phase_Sigma_T_Q))*(MRF_Q - ((MRF_Q-MRF_P)*temp1)/(temp3)));
  }
}

/*Computes the qGGMRF spatial prior second derivative at delta = 0*/
Real_t Mag_QGGMRF_Spatial_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return MRF_Q/(TomoInputsPtr->Mag_Sigma_S_Q*ScannedObjectPtr->Mag_C_S);
}

Real_t Phase_QGGMRF_Spatial_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return MRF_Q/(TomoInputsPtr->Phase_Sigma_S_Q*ScannedObjectPtr->Phase_C_S);
}

/*Computes the qGGMRF spatial prior second derivative at delta = 0*/
Real_t Mag_QGGMRF_Temporal_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return MRF_Q/(TomoInputsPtr->Mag_Sigma_T_Q*ScannedObjectPtr->Mag_C_T);
}

Real_t Phase_QGGMRF_Temporal_SecondDerivative(ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
  return MRF_Q/(TomoInputsPtr->Phase_Sigma_T_Q*ScannedObjectPtr->Phase_C_T);
}

/*Computes the voxel update and returns it. V is the present value of voxel.
THETA1 and THETA2 are the values used in voxel update. Spatial_Nhood and Time_Nhood gives the
values of voxels in the neighborhood of V. Time_BDFlag and Spatial_BDFlag are masks which determine
whether a neighbor should be included in the neighorhood or not.*/
Real_t Mag_FunctionalSubstitution(Real_t V, Real_t THETA1, Real_t THETA2, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Spatial_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood[NHOOD_TIME_MAXDIM-1], bool Spatial_BDFlag[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], bool Time_BDFlag[NHOOD_TIME_MAXDIM-1])
{
  Real_t u,temp1=0,temp2=0,temp_const,RefValue=0,Delta0;
  Real_t QGGMRF_Params;
  int32_t i,j,k;
  RefValue = V;
  /*Need to Loop this for multiple iterations of substitute function*/
  for (i=0; i < NHOOD_Y_MAXDIM; i++)
  for (j=0; j < NHOOD_X_MAXDIM; j++)
  for (k=0; k < NHOOD_Z_MAXDIM; k++)
  {
    if(Spatial_BDFlag[i][j][k] == true && (i != (NHOOD_Y_MAXDIM-1)/2 || j != (NHOOD_X_MAXDIM-1)/2 || k != (NHOOD_Z_MAXDIM-1)/2))
    {
      Delta0 = (RefValue - Spatial_Nhood[i][j][k]);
      if(Delta0 != 0)
      QGGMRF_Params = Mag_QGGMRF_Spatial_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      else {
        QGGMRF_Params = Mag_QGGMRF_Spatial_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);
      }
      temp_const = TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params;
      temp1 += temp_const*Spatial_Nhood[i][j][k];
      temp2 += temp_const;
    }
  }
  for (i=0; i < NHOOD_TIME_MAXDIM - 1; i++)
  {
    if(Time_BDFlag[i] == true)
    {
      Delta0 = (RefValue - Time_Nhood[i]);
      if(Delta0 != 0)
      QGGMRF_Params = Mag_QGGMRF_Temporal_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      else {
        QGGMRF_Params = Mag_QGGMRF_Temporal_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);
      }
      
      temp_const = TomoInputsPtr->Time_Filter[0]*QGGMRF_Params;
      temp1 += temp_const*Time_Nhood[i];
      temp2 += temp_const;
    }
  }
  
  u=(temp1+ (THETA2*V) - THETA1)/(temp2 + THETA2);
  
  RefValue = RefValue + TomoInputsPtr->alpha*(u-RefValue);
  #ifdef POSITIVITY_CONSTRAINT
  if (RefValue <= 0)
  RefValue = 0;
  #endif
  return RefValue;
}


Real_t Phase_FunctionalSubstitution(Real_t V, Real_t THETA1, Real_t THETA2, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Spatial_Nhood[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood[NHOOD_TIME_MAXDIM-1], bool Spatial_BDFlag[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], bool Time_BDFlag[NHOOD_TIME_MAXDIM-1])
{
  Real_t u,temp1=0,temp2=0,temp_const,RefValue=0,Delta0;
  Real_t QGGMRF_Params;
  int32_t i,j,k;
  RefValue = V;
  /*Need to Loop this for multiple iterations of substitute function*/
  for (i=0; i < NHOOD_Y_MAXDIM; i++)
  for (j=0; j < NHOOD_X_MAXDIM; j++)
  for (k=0; k < NHOOD_Z_MAXDIM; k++)
  {
    if(Spatial_BDFlag[i][j][k] == true && (i != (NHOOD_Y_MAXDIM-1)/2 || j != (NHOOD_X_MAXDIM-1)/2 || k != (NHOOD_Z_MAXDIM-1)/2))
    {
      Delta0 = (RefValue - Spatial_Nhood[i][j][k]);
      if(Delta0 != 0)
      QGGMRF_Params = Phase_QGGMRF_Spatial_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      else {
        QGGMRF_Params = Phase_QGGMRF_Spatial_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);
      }
      temp_const = TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params;
      temp1 += temp_const*Spatial_Nhood[i][j][k];
      temp2 += temp_const;
    }
  }
  for (i=0; i < NHOOD_TIME_MAXDIM - 1; i++)
  {
    if(Time_BDFlag[i] == true)
    {
      Delta0 = (RefValue - Time_Nhood[i]);
      if(Delta0 != 0)
      QGGMRF_Params = Phase_QGGMRF_Temporal_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      else {
        QGGMRF_Params = Phase_QGGMRF_Temporal_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);
      }
      
      temp_const = TomoInputsPtr->Time_Filter[0]*QGGMRF_Params;
      temp1 += temp_const*Time_Nhood[i];
      temp2 += temp_const;
    }
  }
  
  u=(temp1+ (THETA2*V) - THETA1)/(temp2 + THETA2);
  
  RefValue = RefValue + TomoInputsPtr->alpha*(u-RefValue);
  #ifdef POSITIVITY_CONSTRAINT
  if (RefValue <= 0)
  RefValue = 0;
  #endif
  return RefValue;
}







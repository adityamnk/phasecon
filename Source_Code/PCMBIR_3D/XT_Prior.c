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

void Mult_2DMatMat(Real_t out[2][2], Real_t in_1[2][2], Real_t in_2[2][2])
{
	int i, j, k;
	for (i = 0; i < 2; i++)
	for (j = 0; j < 2; j++)
	{
		out[i][j] = 0;
		for (k = 0; k < 2; k++)
			out[i][j] += in_1[i][k]*in_2[k][j];
	}
}

void Mult_2DTranMatMat(Real_t out[2][2], Real_t in_1[2][2], Real_t in_2[2][2])
{
	int i, j, k;
	for (i = 0; i < 2; i++)
	for (j = 0; j < 2; j++)
	{
		out[i][j] = 0;
		for (k = 0; k < 2; k++)
			out[i][j] += in_1[k][i]*in_2[k][j];
	}
}

/*Computes the voxel update and returns it. V is the present value of voxel.
THETA1 and THETA2 are the values used in voxel update. Spatial_Nhood and Time_Nhood gives the
values of voxels in the neighborhood of V. Time_BDFlag and Spatial_BDFlag are masks which determine
whether a neighbor should be included in the neighorhood or not.*/
void FunctionalSubstitution(Real_t* V_delta, Real_t* V_beta, Real_t THETA1_delta, Real_t THETA1_beta, Real_t THETA2_delta, Real_t THETA2_beta, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr, Real_t Spatial_Nhood_delta[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood_delta[NHOOD_TIME_MAXDIM-1], Real_t Spatial_Nhood_beta[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], Real_t Time_Nhood_beta[NHOOD_TIME_MAXDIM-1], bool Spatial_BDFlag[NHOOD_Y_MAXDIM][NHOOD_X_MAXDIM][NHOOD_Z_MAXDIM], bool Time_BDFlag[NHOOD_TIME_MAXDIM-1], Real_t DecorrTran[2][2])
{
  Real_t u_delta, u_beta, temp1[2], temp2[2][2], temp_const[2][2], RefValue_delta, RefValue_beta, Delta0_delta, Delta0_beta, QGGMRF_Params[2][2], det, Delta0;
  int32_t i, j, k;

  temp1[0] = 0; temp1[1] = 0;
  temp2[0][0] = 0; temp2[0][1] = 0;
  temp2[1][0] = 0; temp2[1][1] = 0;

  RefValue_beta = *V_beta;
  RefValue_delta = *V_delta;

  /*Need to Loop this for multiple iterations of substitute function*/
  for (i=0; i < NHOOD_Y_MAXDIM; i++)
  for (j=0; j < NHOOD_X_MAXDIM; j++)
  for (k=0; k < NHOOD_Z_MAXDIM; k++)
  {
  	QGGMRF_Params[0][0] = 0; QGGMRF_Params[0][1] = 0;
  	QGGMRF_Params[1][0] = 0; QGGMRF_Params[1][1] = 0;
    	if(Spatial_BDFlag[i][j][k] == true && (i != (NHOOD_Y_MAXDIM-1)/2 || j != (NHOOD_X_MAXDIM-1)/2 || k != (NHOOD_Z_MAXDIM-1)/2))
    	{
      		Delta0_delta = (RefValue_delta - Spatial_Nhood_delta[i][j][k]);
      		Delta0_beta = (RefValue_beta - Spatial_Nhood_beta[i][j][k]);
		Delta0 = DecorrTran[0][0]*Delta0_delta + DecorrTran[0][1]*Delta0_beta; 
      		if(Delta0 != 0)
			QGGMRF_Params[0][0] = Phase_QGGMRF_Spatial_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      		else 
       	     		QGGMRF_Params[0][0] = Phase_QGGMRF_Spatial_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);

		Delta0 = DecorrTran[1][0]*Delta0_delta + DecorrTran[1][1]*Delta0_beta; 
      		if(Delta0 != 0)
	     		QGGMRF_Params[1][1] = Mag_QGGMRF_Spatial_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      		else 
       	     		QGGMRF_Params[1][1] = Mag_QGGMRF_Spatial_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);
     
      		Mult_2DMatMat(temp_const, QGGMRF_Params, DecorrTran);
      		Mult_2DTranMatMat(QGGMRF_Params, DecorrTran, temp_const);
      		temp1[0] += TomoInputsPtr->Spatial_Filter[i][j][k]*(QGGMRF_Params[0][0]*Spatial_Nhood_delta[i][j][k] + QGGMRF_Params[0][1]*Spatial_Nhood_beta[i][j][k]);
      		temp1[1] += TomoInputsPtr->Spatial_Filter[i][j][k]*(QGGMRF_Params[1][0]*Spatial_Nhood_delta[i][j][k] + QGGMRF_Params[1][1]*Spatial_Nhood_beta[i][j][k]);
      		temp2[0][0] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params[0][0];
      		temp2[0][1] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params[0][1];
      		temp2[1][0] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params[1][0];
      		temp2[1][1] += TomoInputsPtr->Spatial_Filter[i][j][k]*QGGMRF_Params[1][1];
    	}
  }
  
  for (i=0; i < NHOOD_TIME_MAXDIM - 1; i++)
  {
  	QGGMRF_Params[0][0] = 0; QGGMRF_Params[0][1] = 0;
  	QGGMRF_Params[1][0] = 0; QGGMRF_Params[1][1] = 0;
    	if(Time_BDFlag[i] == true)
    	{
      		Delta0_delta = (RefValue_delta - Time_Nhood_delta[i]);
      		Delta0_beta = (RefValue_beta - Time_Nhood_beta[i]);
		Delta0 = DecorrTran[0][0]*Delta0_delta + DecorrTran[0][1]*Delta0_beta; 
      		if(Delta0 != 0)
      			QGGMRF_Params[0][0] = Phase_QGGMRF_Temporal_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      		else 
        		QGGMRF_Params[0][0] = Phase_QGGMRF_Temporal_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);
      		
		Delta0 = DecorrTran[1][0]*Delta0_delta + DecorrTran[1][1]*Delta0_beta; 
      		if(Delta0 != 0)
      			QGGMRF_Params[1][1] = Mag_QGGMRF_Temporal_Derivative(Delta0,ScannedObjectPtr,TomoInputsPtr)/(Delta0);
      		else 
        		QGGMRF_Params[1][1] = Mag_QGGMRF_Temporal_SecondDerivative(ScannedObjectPtr,TomoInputsPtr);
      
      		Mult_2DMatMat(temp_const, QGGMRF_Params, DecorrTran);
      		Mult_2DTranMatMat(QGGMRF_Params, DecorrTran, temp_const);

      		temp1[0] += TomoInputsPtr->Time_Filter[0]*(QGGMRF_Params[0][0]*Time_Nhood_delta[i] + QGGMRF_Params[0][1]*Time_Nhood_beta[i]);
      		temp1[1] += TomoInputsPtr->Time_Filter[0]*(QGGMRF_Params[1][0]*Time_Nhood_delta[i] + QGGMRF_Params[1][1]*Time_Nhood_beta[i]);
      		temp2[0][0] += TomoInputsPtr->Time_Filter[0]*QGGMRF_Params[0][0];
      		temp2[0][1] += TomoInputsPtr->Time_Filter[0]*QGGMRF_Params[0][1];
      		temp2[1][0] += TomoInputsPtr->Time_Filter[0]*QGGMRF_Params[1][0];
      		temp2[1][1] += TomoInputsPtr->Time_Filter[0]*QGGMRF_Params[1][1];
    	}
  }

  temp_const[0][0] = THETA2_delta + temp2[0][0]; 
  temp_const[0][1] = temp2[0][1]; 
  temp_const[1][0] = temp2[1][0]; 
  temp_const[1][1] = THETA2_beta + temp2[1][1];

  det = 1.0/(temp_const[0][0]*temp_const[1][1] - temp_const[1][0]*temp_const[0][1]);
  QGGMRF_Params[0][0] = det*temp_const[1][1];
  QGGMRF_Params[1][1] = det*temp_const[0][0];
  QGGMRF_Params[0][1] = -det*temp_const[0][1];
  QGGMRF_Params[1][0] = -det*temp_const[1][0];
 
  u_delta = (temp1[0] + (THETA2_delta*RefValue_delta) - THETA1_delta);
  u_beta = (temp1[1] + (THETA2_beta*RefValue_beta) - THETA1_beta);
  *V_delta = QGGMRF_Params[0][0]*u_delta + QGGMRF_Params[0][1]*u_beta; 
  *V_beta = QGGMRF_Params[1][0]*u_delta + QGGMRF_Params[1][1]*u_beta; 
 
  *V_delta = RefValue_delta + TomoInputsPtr->alpha*(*V_delta - RefValue_delta);
  *V_beta = RefValue_beta + TomoInputsPtr->alpha*(*V_beta - RefValue_beta);
  #ifdef POSITIVITY_CONSTRAINT
  if (*V_delta < 0)
	*V_delta = 0;
  if (*V_beta < 0)
	*V_beta = 0;
  #endif
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










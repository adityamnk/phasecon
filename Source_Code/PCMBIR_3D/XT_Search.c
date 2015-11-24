#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "XT_Constants.h"
#include "XT_Structures.h"

Real_t Cost_NMS (Real_t zr, Real_t zi, Real_t ar, Real_t ai, Real_t br, Real_t bi, Real_t mu, Real_t nu)
{
	Real_t cost = 0, expval, cosval, sinval;

	expval = exp(-zr);
	cosval = expval*cos(-zi);
	sinval = expval*sin(-zi);

	cost = ((zr - ar)*(zr - ar) + (zi - ai)*(zi - ai))*mu;
	cost += ((cosval - br)*(cosval - br) + (sinval - bi)*(sinval - bi))*nu;
	cost /= 2;

	if (zr < 0 || zi < 0)
		cost = INFINITE_COST; 
	return(cost);
}


void NMS_Accept(Real_t zr_acc, Real_t zi_acc, Real_t cost_acc, Real_arr_t* zr, Real_arr_t* zi, Real_t* cost1, Real_t* cost2, Real_t* cost3)
{
	Real_t temp;

	*cost3 = cost_acc;
	zr[3] = zr_acc;
	zi[3] = zi_acc;
	
	if (*cost3 < *cost2)
	{
		temp = zr[2]; zr[2] = zr[3]; zr[3] = temp; 
		temp = zi[2]; zi[2] = zi[3]; zi[3] = temp; 
		temp = *cost2; *cost2 = *cost3; *cost3 = temp; 
	}
	
	if (*cost2 < *cost1)
	{
		temp = zr[1]; zr[1] = zr[2]; zr[2] = temp; 
		temp = zi[1]; zi[1] = zi[2]; zi[2] = temp; 
		temp = *cost1; *cost1 = *cost2; *cost2 = temp; 
	}
}

void NMS_Sort(Real_arr_t* zr, Real_arr_t* zi, Real_t* cost1, Real_t* cost2, Real_t* cost3)
{
	Real_t temp;
	
	if (*cost3 < *cost2)
	{
		temp = zr[2]; zr[2] = zr[3]; zr[3] = temp; 
		temp = zi[2]; zi[2] = zi[3]; zi[3] = temp; 
		temp = *cost2; *cost2 = *cost3; *cost3 = temp; 
	}
		
	if (*cost2 < *cost1)
	{
		temp = zr[1]; zr[1] = zr[2]; zr[2] = temp; 
		temp = zi[1]; zi[1] = zi[2]; zi[2] = temp; 
		temp = *cost1; *cost1 = *cost2; *cost2 = temp; 
	}
	
	if (*cost3 < *cost2)
	{
		temp = zr[2]; zr[2] = zr[3]; zr[3] = temp; 
		temp = zi[2]; zi[2] = zi[3]; zi[3] = temp; 
		temp = *cost2; *cost2 = *cost3; *cost3 = temp; 
	}
}

void NMS_Shrink(Real_arr_t* zr, Real_arr_t* zi, Real_t* cost1, Real_t* cost2, Real_t* cost3, Real_t ar, Real_t ai, Real_t br, Real_t bi, Real_t NMS_sigma, Real_t mu, Real_t nu)
{
	zr[2] = zr[1] + NMS_sigma*(zr[2] - zr[1]);
	zi[2] = zi[1] + NMS_sigma*(zi[2] - zi[1]);
	*cost2 = Cost_NMS(zr[2], zi[2], ar, ai, br, bi, mu, nu);
	
	zr[3] = zr[1] + NMS_sigma*(zr[3] - zr[1]);
	zi[3] = zi[1] + NMS_sigma*(zi[3] - zi[1]);
	*cost3 = Cost_NMS(zr[3], zi[3], ar, ai, br, bi, mu, nu);

	NMS_Sort(zr, zi, cost1, cost2, cost3);
}


/*Function Name - Nelder_Mead_Simplex_2DMin
 * Description - Minimizes a function in complex coordinates, z, of the form |z-a|^2 + |exp(-z)-b|^2.
 * (zr[0],zr[0]) --> centroid
 * (zr[1],zr[1]) --> lowest cost element
 * (zr[2],zr[2]) --> second lowest cost
 * (zr[3],zr[3]) --> highest cost element*/

int32_t Nelder_Mead_Simplex_2DSearch (Real_arr_t* zr, Real_arr_t* zi, Real_t ar, Real_t ai, Real_t br, Real_t bi, Real_t NMS_rho, Real_t NMS_chi, Real_t NMS_gamma, Real_t NMS_sigma, Real_t NMS_threshold, int32_t NMS_MaxIter, Real_t mu, Real_t nu, FILE* debug_file_ptr)
{
	Real_t cost1, cost2, cost3, cost_ref, cost_exp, cost_out, cost_in;
	Real_t zr_ref, zi_ref, zr_exp, zi_exp, zr_out, zi_out, zr_in, zi_in, zr_ocen, zi_ocen, thresh; 
	int32_t i;

	/*Initialization*/
	zr[2] = 2*zr[1]; zi[2] = zi[1];
	zr[3] = zr[1];   zi[3] = 2*zi[1];

	cost1 = Cost_NMS(zr[1], zi[1], ar, ai, br, bi, mu, nu);
	cost2 = Cost_NMS(zr[2], zi[2], ar, ai, br, bi, mu, nu);
	cost3 = Cost_NMS(zr[3], zi[3], ar, ai, br, bi, mu, nu);

	NMS_Sort(zr, zi, &cost1, &cost2, &cost3);
	zr[0] = (zr[1] + zr[2])/2.0;
	zi[0] = (zi[1] + zi[2])/2.0;

	for (i = 0; i < NMS_MaxIter; i++)
	{	
#ifdef EXTRA_DEBUG_MESSAGES
		printf("******* Iteration = %d, zr = %f,%f,%f,%f; zi = %f,%f,%f,%f ********\n", i, zr[0],zr[1],zr[2],zr[3],zi[0],zi[1],zi[2],zi[3]);	
		printf("------- cost1 = %f, cost2 = %f, cost3 = %f --------\n", cost1, cost2, cost3);
#endif
		/*Reflection*/
		zr_ref = zr[0] + NMS_rho*(zr[0] - zr[3]);
		zi_ref = zi[0] + NMS_rho*(zi[0] - zi[3]);
		cost_ref = Cost_NMS(zr_ref, zi_ref, ar, ai, br, bi, mu, nu);
		if (cost_ref < cost2 && cost_ref >= cost1)
		{
			NMS_Accept(zr_ref, zi_ref, cost_ref, zr, zi, &cost1, &cost2, &cost3);
/*			printf("Accepting reflected point zr_ref = %f, zi_ref = %f\n", zr_ref, zi_ref);*/
		}
		else if (cost_ref < cost1)
		{
			zr_exp = zr[0] + NMS_chi*(zr_ref - zr[0]);
			zi_exp = zi[0] + NMS_chi*(zi_ref - zi[0]);
			cost_exp = Cost_NMS(zr_exp, zi_exp, ar, ai, br, bi, mu, nu);
			if (cost_exp < cost_ref)
			{	
				NMS_Accept(zr_exp, zi_exp, cost_exp, zr, zi, &cost1, &cost2, &cost3);
/*				printf("Accepting expanded point zr_exp = %f, zi_exp = %f\n", zr_exp, zi_exp);*/
			}
			else
			{
				NMS_Accept(zr_ref, zi_ref, cost_ref, zr, zi, &cost1, &cost2, &cost3);
/*				printf("Accepting reflected point zr_ref = %f, zi_ref = %f after computing expanded point\n", zr_ref, zi_ref);*/
			}
		}
		else if (cost_ref >= cost2 && cost_ref < cost3)
		{
			zr_out = zr[0] + NMS_gamma*(zr_ref - zr[0]);
			zi_out = zi[0] + NMS_gamma*(zi_ref - zi[0]);
			cost_out = Cost_NMS(zr_out, zi_out, ar, ai, br, bi, mu, nu);
			if (cost_out <= cost_ref)
			{
				NMS_Accept(zr_out, zi_out, cost_out, zr, zi, &cost1, &cost2, &cost3);
/*				printf("Accepting outside contraction point zr_out = %f, zi_out = %f\n", zr_out, zi_out);*/
			}
			else
			{
				NMS_Shrink(zr, zi, &cost1, &cost2, &cost3, ar, ai, br, bi, NMS_sigma, mu, nu);
/*				printf("Doing shrinkage. New points are zr = %f,%f,%f,%f; zi = %f,%f,%f,%f\n", zr[0],zr[1],zr[2],zr[3],zi[0],zi[1],zi[2],zi[3]);*/
			}
		}
		else if (cost_ref >= cost3)
		{
			zr_in = zr[0] - NMS_gamma*(zr[0] - zr[3]);	
			zi_in = zi[0] - NMS_gamma*(zi[0] - zi[3]);	
			cost_in = Cost_NMS(zr_in, zi_in, ar, ai, br, bi, mu, nu);
			if (cost_in < cost3)
			{
				NMS_Accept(zr_in, zi_in, cost_in, zr, zi, &cost1, &cost2, &cost3);
	/*			printf("Accepting inside contraction point zr_in = %f, zi_in = %f\n", zr_in, zi_in);*/
			}
			else
			{
				NMS_Shrink(zr, zi, &cost1, &cost2, &cost3, ar, ai, br, bi, NMS_sigma, mu, nu);
	/*			printf("Doing shrinkage. New points are zr = %f,%f,%f,%f; zi = %f,%f,%f,%f\n", zr[0],zr[1],zr[2],zr[3],zi[0],zi[1],zi[2],zi[3]);*/
			}
		}
		else
			fprintf(debug_file_ptr, "ERROR: An iteration of Nelder Mead Simplex failed to find a point with lower cost! cost_ref = %f, cost1 = %f, cost2 = %f, cost3 = %f.\n", cost_ref, cost1, cost2, cost3);
		zr_ocen = zr[0]; zi_ocen = zi[0];
		zr[0] = (zr[1] + zr[2])/2.0;
		zi[0] = (zi[1] + zi[2])/2.0;

		thresh = (fabs(zr_ocen - zr[0]) + fabs(zi_ocen - zi[0]))/(fabs(zr_ocen) + fabs(zi_ocen))*100;
#ifdef EXTRA_DEBUG_MESSAGES
		fprintf(debug_file_ptr, "The convergence threshold attained is %f percent.\n", thresh);
#endif
/*		if (thresh < NMS_threshold && i > 1)
		{
			fprintf(debug_file_ptr, "The convergence threshold attained is %f percent.\n", thresh);
			break;
		}*/
	}

	return (i);
}


/*int32_t Bootstrap_Line_Search (Real_arr_t* zr, Real_arr_t* zi, Real_t ar, Real_t ai, Real_t br, Real_t bi, Real_t threshold, int32_t MaxIter, FILE* debug_file_ptr)
{

}*/

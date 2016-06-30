/* ============================================================================
 * Copyright (c) 2013 K. Aditya Mohan (Purdue University)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * Neither the name of K. Aditya Mohan, Purdue
 * University, nor the names of its contributors may be used
 * to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


#include <stdio.h>
#include "XT_Constants.h"
#include "tiff.h"
#include "allocate.h"
#include "XT_Structures.h"
#include "XT_MPIIO.h"
#include "XT_Debug.h"

/*Appends values in 'img' to a binary file with name filename. dimensions of img are specified by dim1, dim2, dim3 and dim4*/
int32_t Append2Bin (char *filename, size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t datatype_size, void* img, FILE *debug_file_ptr)
{
  FILE *fp;
  char file[100];
  sprintf(file,"%s.bin", filename);
  fp=fopen (file,"ab");
  check_error(fp==NULL, 1, debug_file_ptr, "Cannot open file %s.\n", file);
  fwrite(img,datatype_size,dim1*dim2*dim3*dim4,fp);	
  fclose(fp);
  return (0);
error:
  return (-1);
}

/*Writes values in 'img' to a binary file with name filename. dimensions of img are specified by dim1, dim2, dim3 and dim4*/
int32_t Write2Bin (char *filename, size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t datatype_size, void* img, FILE *debug_file_ptr)
{
  FILE *fp;
  char file[100];
  sprintf(file,"%s.bin", filename);
  fp=fopen (file,"wb");
  check_error(fp==NULL, 1, debug_file_ptr, "Cannot open file %s.\n", file);
  fwrite(img,datatype_size,dim1*dim2*dim3*dim4,fp);	
  fclose(fp);
  return(0);
error:
  return(-1);
}

/*Reads values in 'img' to a binary file with name filename. dimensions of img are specified by dim1, dim2, dim3 and dim4*/
int32_t Read4mBin (char *filename, size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t datatype_size, void* img, FILE *debug_file_ptr)
{
  	char file[100];
	FILE* fp;
	size_t size, result;
  	sprintf(file,"%s.bin", filename);

	size = dim1*dim2*dim3*dim4;
	fp = fopen (file, "rb" );
	check_error(fp == NULL, 1, debug_file_ptr, "Error in reading file %s.\n", file);	
	result = fread (img, datatype_size, size, fp);
  	check_error(result != size, 1, debug_file_ptr, "Number of elements read does not match required, required = %zu, read = %zu\n", size, result);
	fclose(fp);
	return(0);
error:
	if (fp)
		fclose(fp);
	return(-1);
}

/*Writes values in 'img' to tiff files. dimensions of img are specified by height and width.
hounsfield_flag if set, converts all values to HU and scales appropriately before writing to tiff file*/
int32_t Write2Tiff(char* filename, int height, int width, Real_arr_t** img, int scale_flag, FILE *debug_file_ptr)
{
	struct TIFF_img out_img;
	int i,j;
	Real_arr_t pixel;
	Real_arr_t maxpix,minpix,avgpix,scale;
	FILE *fp;
	char file[100];

	get_TIFF ( &out_img, height, width, 'g' );
	avgpix=0;
		
	maxpix=img[0][0];minpix=maxpix;
	for ( i = 0; i < height; i++ )
 	for ( j = 0; j < width; j++ ) {
		maxpix=(maxpix<img[i][j])?img[i][j]:maxpix;
		minpix=(minpix>img[i][j])?img[i][j]:minpix;
		avgpix+=img[i][j];	
	}
	avgpix/=(height*width);
	check_debug(1, debug_file_ptr, "file=%s,maxpix=%e,minpix=%e,avgpix=%e,height=%d,width=%d.\n",filename,maxpix,minpix,avgpix,height,width);

/*	if (scale_flag == 1){
		maxpix = ABSORP_COEF_MAX;
		minpix = ABSORP_COEF_MIN;
	}
	else if (scale_flag == 2)
	{
		maxpix = REF_IND_DEC_MAX;
		minpix = REF_IND_DEC_MIN;
	}*/

	scale=255/(maxpix-minpix);
	 for ( i = 0; i < height; i++ )
 	 for ( j = 0; j < width; j++ ) {
    		pixel = (int32_t)((img[i][j]-minpix)*scale);
    		if(pixel>255) {
      			out_img.mono[i][j] = 255;
    		}
    		else {
      			if(pixel<0) out_img.mono[i][j] = 0;
      			else out_img.mono[i][j] = pixel;
    		}
  	}
	sprintf(file,"%s.tif", filename);	
	if ((fp = fopen(file, "wb")) == NULL)
    		check_error(1, 1, debug_file_ptr, "cannot open file %s.\n",filename);

	/* write image */
	if ( write_TIFF ( fp, &out_img ) )
    		check_error(1, 1, debug_file_ptr, "error writing TIFF file %s.\n", filename );

	free_TIFF (&out_img);
  /* close image file */
  fclose ( fp );
  return(0);
error:
  return(-1);
}

/*Writes to tiff files from a int32 array*/
int32_t WriteInt32Tiff(char* filename, int height, int width, int32_t** imgin, int hounsfield_flag, FILE *debug_file_ptr)
{
	Real_arr_t** img;
	int32_t i, j, flag = 0;

	img = (Real_arr_t**)multialloc(sizeof(Real_arr_t), 2, height, width);
	for (i = 0; i < height; i++)
	for (j = 0; j < width; j++)
		img[i][j] = (Real_arr_t)imgin[i][j];
	flag = Write2Tiff(filename, height, width, img, hounsfield_flag, debug_file_ptr);
	multifree(img,2);
	return(flag);
}

/*Writes a multi-dimension array in 'img' to tiff files with dimension given in dim[4].
dim2loop_1 and dim2loop_2 specifies the dimension over which we loop and write the tiff files.
dim2write_1 and dim2write_2 specifies the dimensions which are written to tiff files.*/
int32_t WriteMultiDimArray2Tiff (char *filename, int dim[4], int dim2loop_1, int dim2loop_2, int dim2write_1, int dim2write_2, Real_arr_t* img, int scale_flag, int dataskip_step, int dataskip_num, FILE* debug_file_ptr)
{
	char file[100];
	Real_arr_t** img_temp;
	int i,j,temp,k,l,flag = 0;

	if (dim2loop_1 > dim2loop_2){
		temp = dim2loop_2;
		dim2loop_2 = dim2loop_1;
		dim2loop_1 = temp;
		check_warn(1, debug_file_ptr, "Swapping variables dim2write_1 and dim2write_2\n");
	}
 		
	check_debug(1, debug_file_ptr, "Writing to tiff file %s*.tif \n", filename);
	img_temp = (Real_arr_t**) multialloc(sizeof(Real_arr_t), 2, dim[dim2write_1], dim[dim2write_2]);
	for (i=0; i<dim[dim2loop_1]; i++){
	for (j=0; j<dim[dim2loop_2]; j++){
	for (k=0; k<dim[dim2write_1]; k++){
	for (l=0; l<dim[dim2write_2]; l++){
		if (dim2write_1 == 2 && dim2write_2 == 3)
			img_temp[k][l] = img[(((i*dim[1]+j)*dim[2]+k)*dim[3]+l)*dataskip_num + dataskip_step];
		else if (dim2write_1 == 1 && dim2write_2 == 3)
			img_temp[k][l] = img[(((i*dim[1]+k)*dim[2]+j)*dim[3]+l)*dataskip_num + dataskip_step]; 	
		else if (dim2write_1 == 0 && dim2write_2 == 3)
			img_temp[k][l] = img[(((k*dim[1]+i)*dim[2]+j)*dim[3]+l)*dataskip_num + dataskip_step]; 
		else if (dim2write_1 == 1 && dim2write_2 == 2)
			img_temp[k][l] = img[(((i*dim[1]+k)*dim[2]+l)*dim[3]+j)*dataskip_num + dataskip_step]; 	
		else if (dim2write_1 == 0 && dim2write_2 == 2)
			img_temp[k][l] = img[(((k*dim[1]+i)*dim[2]+l)*dim[3]+j)*dataskip_num + dataskip_step]; 	
		else if (dim2write_1 == 0 && dim2write_2 == 1)
			img_temp[k][l] = img[(((k*dim[1]+l)*dim[2]+i)*dim[3]+j)*dataskip_num + dataskip_step]; 
		else
			sentinel(1, debug_file_ptr, "Dimensions not recognized.\n");
	}}
		sprintf(file, "%s_d1_%d_d2_%d",filename,i,j);
		if (Write2Tiff(file, dim[dim2write_1], dim[dim2write_2], img_temp, scale_flag, debug_file_ptr)) flag = -1;
	}
	}

	multifree (img_temp,2);
	return(flag);
error:
	multifree (img_temp,2);
	return (-1);
}

int32_t write_ObjectProjOff2TiffBinPerIter (Sinogram* SinogramPtr, ScannedObject* ScannedObjectPtr, TomoInputs* TomoInputsPtr)
{
	int32_t flag = 0;
	int dimTiff[4];
	
	char object_file[100];

	/*	size = ScannedObjectPtr->N_z*ScannedObjectPtr->N_y*ScannedObjectPtr->N_x;*/
		Write2Bin (MAGNETIZATION_FILENAME, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, 3, sizeof(Real_arr_t), &(ScannedObjectPtr->Magnetization[0][0][0][0]), TomoInputsPtr->debug_file_ptr);
		Write2Bin (MAGVECPOT_FILENAME, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, 3, sizeof(Real_arr_t), &(ScannedObjectPtr->MagPotentials[0][0][0][0]), TomoInputsPtr->debug_file_ptr);
/*		if (write_SharedBinFile_At (MAGNETIZATION_FILENAME, &(ScannedObjectPtr->Magnetization[0][0][0][0]), TomoInputsPtr->node_rank*size*3, size*3, TomoInputsPtr->debug_file_ptr)) flag = -1;
		if (write_SharedBinFile_At (ELECCHARGEDENSITY_FILENAME, &(ScannedObjectPtr->ChargeDensity[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;*/
#ifdef VFET_ELEC_RECON
		Write2Bin (ELECCHARGEDENSITY_FILENAME, 1, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, sizeof(Real_arr_t), &(ScannedObjectPtr->ChargeDensity[0][0][0]), TomoInputsPtr->debug_file_ptr);
		Write2Bin (ELECPOT_FILENAME, 1, ScannedObjectPtr->N_z, ScannedObjectPtr->N_y, ScannedObjectPtr->N_x, sizeof(Real_arr_t), &(ScannedObjectPtr->ElecPotentials[0][0][0]), TomoInputsPtr->debug_file_ptr);
#endif
/*		if (write_SharedBinFile_At (MAGVECPOT_FILENAME, &(ScannedObjectPtr->MagPotentials[0][0][0][0]), TomoInputsPtr->node_rank*size*3, size*3, TomoInputsPtr->debug_file_ptr)) flag = -1;
		if (write_SharedBinFile_At (ELECPOT_FILENAME, &(ScannedObjectPtr->ElecPotentials[0][0][0]), TomoInputsPtr->node_rank*size, size, TomoInputsPtr->debug_file_ptr)) flag = -1;*/
		if (TomoInputsPtr->Write2Tiff == 1)
		{
				dimTiff[0] = ScannedObjectPtr->N_z; dimTiff[1] = ScannedObjectPtr->N_y; dimTiff[2] = ScannedObjectPtr->N_x; dimTiff[3] = 3;		
				sprintf (object_file, "%s_n%d", MAGNETIZATION_FILENAME, TomoInputsPtr->node_rank);
				if(WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 3, 1, 2, &(ScannedObjectPtr->Magnetization[0][0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) flag = -1;
				dimTiff[0] = ScannedObjectPtr->N_z; dimTiff[1] = ScannedObjectPtr->N_y; dimTiff[2] = ScannedObjectPtr->N_x; dimTiff[3] = 3;		
				sprintf (object_file, "%s_n%d", MAGVECPOT_FILENAME, TomoInputsPtr->node_rank);
				if(WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 3, 1, 2, &(ScannedObjectPtr->MagPotentials[0][0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) flag = -1;
#ifdef VFET_ELEC_RECON
				dimTiff[0] = 1; dimTiff[1] = ScannedObjectPtr->N_z; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;		
				sprintf (object_file, "%s_n%d", ELECCHARGEDENSITY_FILENAME, TomoInputsPtr->node_rank);
				if(WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->ChargeDensity[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) flag = -1;
				dimTiff[0] = 1; dimTiff[1] = ScannedObjectPtr->N_z; dimTiff[2] = ScannedObjectPtr->N_y; dimTiff[3] = ScannedObjectPtr->N_x;		
				sprintf (object_file, "%s_n%d", ELECPOT_FILENAME, TomoInputsPtr->node_rank);
				if(WriteMultiDimArray2Tiff (object_file, dimTiff, 0, 1, 2, 3, &(ScannedObjectPtr->ElecPotentials[0][0][0]), 0, 0, 1, TomoInputsPtr->debug_file_ptr)) flag = -1;
#endif

		}
			/*Changed above line so that output image is scaled from min to max*/

	return (flag);
}

/*Writes boolean array to tif files*/
int32_t WriteBoolArray2Tiff (char *filename, int dim[4], int dim2loop_1, int dim2loop_2, int dim2write_1, int dim2write_2, bool* imgin, int hounsfield_flag, FILE* debug_file_ptr)
{
	Real_arr_t* img;
	int32_t i, flag = 0;

	img = (Real_arr_t*)get_spc(dim[0]*dim[1]*dim[2]*dim[3], sizeof(Real_arr_t));
	for (i = 0; i < dim[0]*dim[1]*dim[2]*dim[3]; i++)
	{
		if (imgin[i] == true)
			img[i] = 1;
		else
			img[i] = 0;
	}

	flag = WriteMultiDimArray2Tiff (filename, dim, dim2loop_1, dim2loop_2, dim2write_1, dim2write_2, img, hounsfield_flag, 0, 1, debug_file_ptr);
	free(img);
	return (flag);
}

void Arr1DToArr2D (Real_arr_t* arr1d, Real_arr_t** arr2d, int32_t N_y, int32_t N_x)
{
	int32_t i;
	arr2d = (Real_arr_t**)multialloc(sizeof(Real_arr_t*),1,N_y); 
	for (i = 0; i < N_y; i++)
		arr2d[i] = arr1d + i*N_x;
}

void Arr2DToArr1D (Real_arr_t* arr1d, Real_arr_t** arr2d)
{
	arr1d = &(arr2d[0][0]);
	multifree(arr2d, 1);
}

void Arr1DToArr3D (Real_arr_t* arr1d, Real_arr_t*** arr3d, int32_t N_z, int32_t N_y, int32_t N_x)
{
	int32_t i, j;
	arr3d = (Real_arr_t***)multialloc(sizeof(Real_arr_t*),2,N_z,N_y); 
	for (i = 0; i < N_z; i++)
	for (j = 0; j < N_y; j++)
		arr3d[i][j] = arr1d + i*N_y*N_x + j*N_x;
}

void Arr3DToArr1D (Real_arr_t* arr1d, Real_arr_t*** arr3d)
{
	arr1d = &(arr3d[0][0][0]);
	multifree(arr3d, 2);
}

Real_arr_t**** Arr1DToArr4D (Real_arr_t* arr1d, int32_t N_t, int32_t N_z, int32_t N_y, int32_t N_x)
{
	int32_t i, j, k;
	Real_arr_t**** arr4d = (Real_arr_t****)multialloc(sizeof(Real_arr_t*),3,N_t,N_z,N_y); 
	for (i = 0; i < N_t; i++)
	for (j = 0; j < N_z; j++)
	for (k = 0; k < N_y; k++)
		arr4d[i][j][k] = arr1d + i*N_z*N_y*N_x + j*N_y*N_x + k*N_x;
	return (arr4d);
}

Real_arr_t* Arr4DToArr1D (Real_arr_t**** arr4d)
{
	Real_arr_t* arr1d = &(arr4d[0][0][0][0]);
	multifree(arr4d, 3);
	return (arr1d);
}

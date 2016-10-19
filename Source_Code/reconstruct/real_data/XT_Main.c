#include <stdio.h>
#include <stdint.h>
/*#include <mpi.h>*/
#include <stdlib.h>
#include <getopt.h>
#include "XT_Main.h"
#include "vfetmbir4d.h"

/*Function prototype definitions which will be defined later in the file.*/
void read_data (float **data_unflip_x, float **data_unflip_y, float **proj_angles_x, float **proj_angles_y, int32_t proj_rows, int32_t proj_cols, int32_t proj_x_num, int32_t proj_y_num, float vox_wid, FILE* debug_file_ptr);
void read_command_line_args (int32_t argc, char **argv, int32_t *proj_rows, int32_t *proj_cols, int32_t *proj_x_num, int32_t *proj_y_num, int32_t *x_widnum, int32_t *y_widnum, int32_t *z_widnum, float *vox_wid, float *qggmrf_sigma, float *qggmrf_c, float *convg_thresh, float *admm_mu, int32_t *admm_maxiters, float *data_var, uint8_t *restart, FILE* debug_msg_ptr);

/*The main function which reads the command line arguments, reads the data,
  and does the reconstruction.*/
int main(int argc, char **argv)
{
	uint8_t restart;
	int32_t proj_rows, proj_cols, proj_x_num, proj_y_num, nodes_num, nodes_rank, admm_maxiters, x_widnum, y_widnum, z_widnum;
	float *magobject, *elecobject, *data_unflip_x, *data_unflip_y, *proj_angles_x, *proj_angles_y, vox_wid, qggmrf_sigma, qggmrf_c, elec_sigma, elec_c, convg_thresh, admm_mu, data_var;
	FILE *debug_msg_ptr;

	/*initialize MPI process.*/	
	/*MPI_Init(&argc, &argv);*/
	/*Find the total number of nodes.*/
	/*MPI_Comm_size(MPI_COMM_WORLD, &nodes_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &nodes_rank);*/
	nodes_num = 1; nodes_rank = 0;
	
	/*All messages to help debug any potential mistakes or bugs are written to debug.log*/
	debug_msg_ptr = fopen("debug.log", "w");
	debug_msg_ptr = stdout;	
	/*Read the command line arguments to determine the reconstruction parameters*/
	read_command_line_args (argc, argv, &proj_rows, &proj_cols, &proj_x_num, &proj_y_num, &x_widnum, &y_widnum, &z_widnum, &vox_wid, &qggmrf_sigma, &qggmrf_c, &convg_thresh, &admm_mu, &admm_maxiters, &data_var, &restart, debug_msg_ptr);
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Number of nodes is %d and command line input argument values are proj_rows = %d, proj_cols = %d, proj_x_num = %d, proj_y_num = %d, x_widnum = %d, y_widnum = %d, z_widnum = %d, vox_wid = %e, qggmrf_sigma = %e, qggmrf_c = %e, convg_thresh = %e, admm mu = %e, admm maxiters = %d, data variance = %f, restart = %d\n", nodes_num, proj_rows, proj_cols, proj_x_num, proj_y_num, x_widnum, y_widnum, z_widnum, vox_wid, qggmrf_sigma, qggmrf_c, convg_thresh, admm_mu, admm_maxiters, data_var, restart);	
	
	/*Allocate memory for data arrays used for reconstruction.*/
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Allocating memory for data ....\n");

	/*Read data*/
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Reading data ....\n");
	read_data (&data_unflip_x, &data_unflip_y, &proj_angles_x, &proj_angles_y, proj_rows, proj_cols, proj_x_num, proj_y_num, vox_wid, debug_msg_ptr);
	
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Reconstructing the data ....\n");
	/*Run the reconstruction*/
	vfet_reconstruct (&magobject, data_unflip_x, data_unflip_y, proj_angles_x, proj_angles_y, proj_rows, proj_cols, proj_x_num, proj_y_num, x_widnum, y_widnum, z_widnum, vox_wid, qggmrf_sigma, qggmrf_c, convg_thresh, admm_mu, admm_maxiters, data_var, restart, debug_msg_ptr);
	/*free(magobject);
	free(elecobject);*/
	
	free(data_unflip_x);
	free(data_unflip_y);
	free(proj_angles_x);
	free(proj_angles_y);

	fclose (debug_msg_ptr); 
	/*MPI_Finalize();*/
	return (0);
}

void read_BinFile (char filename[100], float* data, size_t offset, size_t size, FILE* debug_file_ptr)
{
  	char file[100];
	FILE* fp;
	size_t result;
  	sprintf(file,"%s.bin", filename);

	fp = fopen (file, "rb" );
	if(fp == NULL)
	{
		fprintf(debug_file_ptr, "Error in reading file %s.\n", file);
		exit(-1);
	}

	result = fseek(fp, offset*sizeof(float), SEEK_SET);

	if (result != 0)
	{
		fprintf(debug_file_ptr, "Could not seek the specified offset in file %s\n", file);
	}

	result = fread (data, sizeof(float), size, fp);

  	if(result != size)
	{
		fprintf(debug_file_ptr, "Number of elements read does not match required, required = %zu, read = %zu\n", size, result);
		exit(-1);
	}
	fclose(fp);
}

void write_BinFile (char filename[100], float* data, size_t size, FILE* debug_file_ptr)
{
  	char file[100];
	FILE* fp;
	size_t result;
  	sprintf(file,"%s.bin", filename);

	fp = fopen (file, "wb" );
	if(fp == NULL)
	{
		fprintf(debug_file_ptr, "Error in reading file %s.\n", file);
		exit(-1);
	}

  	result = fwrite(data, sizeof(float), size, fp);	

  	if(result != size)
	{
		fprintf(debug_file_ptr, "Number of elements written does not match total, total = %zu, written = %zu\n", size, result);
		exit(-1);
	}
	fclose(fp);
}

void read_data (float **data_unflip_x, float **data_unflip_y, float **proj_angles_x, float **proj_angles_y, int32_t proj_rows, int32_t proj_cols, int32_t proj_x_num, int32_t proj_y_num, float vox_wid, FILE* debug_file_ptr)
{
	char data_unflip_x_filename[] = DATA_UNFLIP_X_FILENAME;
	char data_unflip_y_filename[] = DATA_UNFLIP_Y_FILENAME;
	char proj_angles_x_filename[] = PROJ_ANGLES_X_FILENAME;
	char proj_angles_y_filename[] = PROJ_ANGLES_Y_FILENAME;
	int32_t offset, size;
	int32_t i, idx, rank, num_nodes;

/*	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);*/
	rank = 0; num_nodes = 1;
	*proj_angles_x = (float*)calloc (proj_x_num, sizeof(float));
	*proj_angles_y = (float*)calloc (proj_y_num, sizeof(float));
	
	*data_unflip_x = (float*)calloc ((proj_x_num*proj_rows*proj_cols)/num_nodes, sizeof(float));
	*data_unflip_y = (float*)calloc ((proj_y_num*proj_rows*proj_cols)/num_nodes, sizeof(float));

/*	idx = 0;	
	for (i = -(proj_num-1); i <= (proj_num-1); i=i+2)
	{
		(*proj_angles)[idx] = M_PI*((float)(i))/180;
		idx++;
	}
	
	vfettomo_forward_project (data_unflip_x, data_unflip_y, *proj_angles, proj_rows, proj_cols, proj_num, vox_wid, debug_file_ptr);
	
        size = proj_rows*proj_cols*proj_num;
	write_BinFile (proj_angles_filename, *proj_angles, proj_num, debug_file_ptr);
	write_BinFile (data_unflip_x_filename, *data_unflip_x, size, debug_file_ptr);
	write_BinFile (data_unflip_y_filename, *data_unflip_y, size, debug_file_ptr);*/
	
	read_BinFile (proj_angles_x_filename, *proj_angles_x, 0, proj_x_num, debug_file_ptr);
	read_BinFile (proj_angles_y_filename, *proj_angles_y, 0, proj_y_num, debug_file_ptr);
	size = proj_rows*proj_cols/num_nodes;
	for (i = 0; i < proj_x_num; i++)
	{
		offset = i*proj_rows*proj_cols + rank*size;
		read_BinFile (data_unflip_x_filename, *data_unflip_x + i*size, offset, size, debug_file_ptr);
	}

	for (i = 0; i < proj_y_num; i++)
	{
		offset = i*proj_rows*proj_cols + rank*size;
		read_BinFile (data_unflip_y_filename, *data_unflip_y + i*size, offset, size, debug_file_ptr);
	}
}

void read_command_line_args (int32_t argc, char **argv, int32_t *proj_rows, int32_t *proj_cols, int32_t *proj_x_num, int32_t *proj_y_num, int32_t *x_widnum, int32_t *y_widnum, int32_t *z_widnum, float *vox_wid, float *qggmrf_sigma, float *qggmrf_c, float *convg_thresh, float *admm_mu, int32_t *admm_maxiters, float *data_var, uint8_t *restart, FILE* debug_msg_ptr)
/*Function which parses the command line input to the C code and initializes several variables.*/
{
	int32_t option_index;
	char c;
	static struct option long_options[] =
        {
               {"proj_rows",  required_argument, 0, 'a'}, /*Number of columns in the projection image. Typically, it is the number of detector bins in the cross-axial direction.*/
               {"proj_cols",  required_argument, 0, 'b'}, /*Number of rows (or slices) in the projection image. Typically, it is the number of detector bins in the axial direction.*/
               {"proj_x_num",  required_argument, 0, 'c'}, /*Total number of 2D projections used for reconstruction.*/
               {"vox_wid",  required_argument, 0, 'd'}, /*Side length of a cubic voxel in inverse units of linear attenuation coefficient of the object. 
		For example, if units of "vox_wid" is mm, then attenuation coefficient will have units of mm^-1, and vice versa.
		Note that attenuation coefficient is what we are trying to reconstruct.*/
               {"qggmrf_sigma",  required_argument, 0, 'e'}, /*Spatial regularization parameter of the prior model.*/
               {"qggmrf_c",  required_argument, 0, 'f'}, 
		/*parameter of the spatial qGGMRF prior model. 
 		Should be fixed to be much lesser (typically 0.01 times) than the ratio of voxel difference over an edge to sigma_s.*/ 
               {"convg_thresh",    required_argument, 0, 'g'}, /*Used to determine when the algorithm is converged at each stage of multi-resolution.
		If the ratio of the average magnitude of voxel updates to the average voxel value expressed as a percentage is less
		than "convg_thresh" then the algorithm is assumed to have converged and the algorithm stops.*/
               {"admm_mu",    required_argument, 0, 'h'}, /*ADMM parameter used to control convergence*/
               {"admm_maxiters",    required_argument, 0, 'i'}, /*Maximum number of admm iterations*/
               {"restart",    no_argument, 0, 'j'}, /*If the reconstruction gets killed due to any unfortunate reason (like exceeding walltime in a super-computing cluster), use this flag to restart the reconstruction from the beginning of the current multi-resolution stage. Don't use restart if WRITE_EVERY_ITER  is 1.*/
               {"x_widnum",    required_argument, 0, 'k'}, /*number of pixels to reconstruct in x-axis direction*/
               {"y_widnum",    required_argument, 0, 'l'}, /*number of pixels to reconstruct in y-axis direction*/
               {"z_widnum",    required_argument, 0, 'm'}, /*number of pixels to reconstruct in z-axis direction*/
               {"proj_y_num",  required_argument, 0, 'n'}, /*Total number of 2D projections used for reconstruction.*/
               {"data_variance",  required_argument, 0, 'o'}, /*Total number of 2D projections used for reconstruction.*/
		{0, 0, 0, 0}
         };

	*restart = 0;
	*data_var = 1;
	while(1)
	{		
	   c = getopt_long (argc, argv, "a:b:c:d:e:f:g:h:i:jk:l:m:n:o:", long_options, &option_index);
           /* Detect the end of the options. */
          if (c == -1) break;
	  switch (c) { 
		case  0 : fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Argument not recognized\n");		break;
		case 'a': *proj_rows = (int32_t)atoi(optarg);			break;
		case 'b': *proj_cols = (int32_t)atoi(optarg);			break;
		case 'c': *proj_x_num = (int32_t)atoi(optarg);			break;
		case 'd': *vox_wid = (float)atof(optarg);			break;
		case 'e': *qggmrf_sigma = (float)atof(optarg);			break;
		case 'f': *qggmrf_c = (float)atof(optarg);				break;
		case 'g': *convg_thresh = (float)atof(optarg);			break;
		case 'h': *admm_mu = (float)atof(optarg);			break;
		case 'i': *admm_maxiters = (int32_t)atoi(optarg);			break;
		case 'j': *restart = 1;		break;
		case 'k': *x_widnum = (int32_t)atoi(optarg);			break;
		case 'l': *y_widnum = (int32_t)atoi(optarg);			break;
		case 'm': *z_widnum = (int32_t)atoi(optarg);			break;
		case 'n': *proj_y_num = (int32_t)atoi(optarg);			break;
		case 'o': *data_var = (float)atof(optarg);			break;
		case '?': fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Cannot recognize argument %s\n",optarg); break;
		}
	}

	if(argc-optind > 0)
		fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Argument list has an error\n");
}


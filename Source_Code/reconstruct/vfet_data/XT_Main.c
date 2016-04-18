#include <stdio.h>
#include <stdint.h>
#include <mpi.h>
#include <stdlib.h>
#include <getopt.h>
#include "XT_Main.h"
#include "vfetmbir4d.h"

/*Function prototype definitions which will be defined later in the file.*/
void read_data (float **data_unflip_x, float **data_flip_x, float **data_unflip_y, float **data_flip_y, float **proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, float vox_wid, float rot_center, FILE* debug_file_ptr);
void read_command_line_args (int32_t argc, char **argv, int32_t *proj_rows, int32_t *proj_cols, int32_t *proj_num, float *vox_wid, float *rot_center, float *mag_sigma, float *mag_c, float *elec_sigma, float *elec_c, float *convg_thresh, uint8_t *restart, FILE* debug_msg_ptr);

/*The main function which reads the command line arguments, reads the data,
  and does the reconstruction.*/
int main(int argc, char **argv)
{
	uint8_t restart;
	int32_t proj_rows, proj_cols, proj_num, nodes_num, nodes_rank;
	float *magobject, *elecobject, *data_unflip_x, *data_flip_x, *data_unflip_y, *data_flip_y, *proj_angles, vox_wid, rot_center, mag_sigma, mag_c, elec_sigma, elec_c, convg_thresh;
	FILE *debug_msg_ptr;

	/*initialize MPI process.*/	
	MPI_Init(&argc, &argv);
	/*Find the total number of nodes.*/
	MPI_Comm_size(MPI_COMM_WORLD, &nodes_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &nodes_rank);
	
	/*All messages to help debug any potential mistakes or bugs are written to debug.log*/
	debug_msg_ptr = fopen("debug.log", "w");
	debug_msg_ptr = stdout;	
	/*Read the command line arguments to determine the reconstruction parameters*/
	read_command_line_args (argc, argv, &proj_rows, &proj_cols, &proj_num, &vox_wid, &rot_center, &mag_sigma, &mag_c, &elec_sigma, &elec_c, &convg_thresh, &restart, debug_msg_ptr);
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Number of nodes is %d and command line input argument values are proj_rows = %d, proj_cols = %d, proj_num = %d, vox_wid = %e, rot_center = %e, mag_sigma = %e, mag_c = %e, elec_sigma = %e, elec_c = %e, convg_thresh = %e, restart = %d\n", nodes_num, proj_rows, proj_cols, proj_num, vox_wid, rot_center, mag_sigma, mag_c, elec_sigma, elec_c, convg_thresh, restart);	
	
	/*Allocate memory for data arrays used for reconstruction.*/
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Allocating memory for data ....\n");

	/*Read data*/
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Reading data ....\n");
	read_data (&data_unflip_x, &data_flip_x, &data_unflip_y, &data_flip_y, &proj_angles, proj_rows, proj_cols, proj_num, vox_wid, rot_center, debug_msg_ptr);
	
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Reconstructing the data ....\n");
	/*Run the reconstruction*/
	vfet_reconstruct (&magobject, &elecobject, data_unflip_x, data_flip_x, data_unflip_y, data_flip_y, proj_angles, proj_rows, proj_cols, proj_num, vox_wid, rot_center, mag_sigma, mag_c, elec_sigma, elec_c, convg_thresh, restart, debug_msg_ptr);
	/*free(magobject);
	free(elecobject);*/
	
	free(data_unflip_x);
	free(data_flip_x);
	free(data_unflip_y);
	free(data_flip_y);
	free(proj_angles);

	fclose (debug_msg_ptr); 
	MPI_Finalize();
	return (0);
}

void read_BinFile (char filename[100], float* data, int32_t offset, int32_t size, FILE* debug_file_ptr)
{
	MPI_File fh;
	MPI_Status status;
	char BinFilename[100];
	int32_t len;

    	sprintf(BinFilename, "%s.bin", filename);
	MPI_File_open(MPI_COMM_WORLD, BinFilename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_at(fh, offset*sizeof(float), data, size, MPI_FLOAT, &status);
	MPI_Get_count(&status, MPI_FLOAT, &len);
	MPI_File_close(&fh);
    	if(len == MPI_UNDEFINED || len != size)
	{
		fprintf (debug_file_ptr, "ERROR: read_BinFile: Read %d number of elements from the file %s at an offset of %d bytes.\n. However, required number of elements is %d.", len, filename, offset, size);
		exit(1);
	}
}

void write_BinFile (char filename[100], float* data, int32_t offset, int32_t size, FILE* debug_file_ptr)
{
	MPI_File fhw;
	MPI_Status status;
	char BinFilename[100];
	int32_t len;

    	sprintf(BinFilename, "%s.bin", filename);
	MPI_File_open(MPI_COMM_WORLD, BinFilename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fhw);
	MPI_File_write_at(fhw, offset*sizeof(float), data, size, MPI_FLOAT, &status);
	MPI_Get_count(&status, MPI_FLOAT, &len);
	MPI_File_close(&fhw);
    	if(len == MPI_UNDEFINED || len != size)
	{
		fprintf (debug_file_ptr, "ERROR: write_BinFile: Wrote %d number of elements to the file %s at an offset of %d bytes.\n. However, actual number of elements to be written is %d.", len, filename, offset, size);
		exit(1);
	}
}


void read_data (float **data_unflip_x, float **data_flip_x, float **data_unflip_y, float **data_flip_y, float **proj_angles, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, float vox_wid, float rot_center, FILE* debug_file_ptr)
{
	/*char data_unflip_x_filename[] = DATA_UNFLIP_X_FILENAME;
	char data_flip_x_filename[] = DATA_FLIP_X_FILENAME;
	char data_unflip_y_filename[] = DATA_UNFLIP_Y_FILENAME;
	char data_flip_y_filename[] = DATA_FLIP_Y_FILENAME;
	char proj_angles_filename[] = PROJ_ANGLES_FILENAME;*/
/*	int32_t offset, size;*/
	int32_t i, idx, rank, num_nodes;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
	*proj_angles = (float*)calloc (proj_num, sizeof(float));
	
	*data_unflip_x = (float*)calloc ((proj_num*proj_rows*proj_cols)/num_nodes, sizeof(float));
	*data_flip_x = (float*)calloc ((proj_num*proj_rows*proj_cols)/num_nodes, sizeof(float));
	*data_unflip_y = (float*)calloc ((proj_num*proj_rows*proj_cols)/num_nodes, sizeof(float));
	*data_flip_y = (float*)calloc ((proj_num*proj_rows*proj_cols)/num_nodes, sizeof(float));

	idx = 0;	
	for (i = -(proj_num-1); i <= (proj_num-1); i=i+2)
	{
		(*proj_angles)[idx] = M_PI*((float)(i))/180;
		idx++;
	}
	
	vfettomo_forward_project (data_unflip_x, data_flip_x, data_unflip_y, data_flip_y, *proj_angles, proj_rows, proj_cols, proj_num, vox_wid, rot_center, debug_file_ptr);
	
/*	read_BinFile (proj_angles_filename, *proj_angles, 0, proj_num, debug_file_ptr);*/
/*	size = proj_rows*proj_cols/num_nodes;
	for (i = 0; i < proj_num; i++)
	{
		offset = i*proj_rows*proj_cols + rank*size;
		read_BinFile (data_unflip_x_filename, *data_unflip_x + i*size, offset, size, debug_file_ptr);
		read_BinFile (data_flip_x_filename, *data_flip_x + i*size, offset, size, debug_file_ptr);
		read_BinFile (data_unflip_y_filename, *data_unflip_y + i*size, offset, size, debug_file_ptr);
		read_BinFile (data_flip_y_filename, *data_flip_y + i*size, offset, size, debug_file_ptr);
	}*/
}

void read_command_line_args (int32_t argc, char **argv, int32_t *proj_rows, int32_t *proj_cols, int32_t *proj_num, float *vox_wid, float *rot_center, float *mag_sigma, float *mag_c, float *elec_sigma, float *elec_c, float *convg_thresh, uint8_t *restart, FILE* debug_msg_ptr)
/*Function which parses the command line input to the C code and initializes several variables.*/
{
	int32_t option_index;
	char c;
	static struct option long_options[] =
        {
               {"proj_rows",  required_argument, 0, 'a'}, /*Number of columns in the projection image. Typically, it is the number of detector bins in the cross-axial direction.*/
               {"proj_cols",  required_argument, 0, 'b'}, /*Number of rows (or slices) in the projection image. Typically, it is the number of detector bins in the axial direction.*/
               {"proj_num",  required_argument, 0, 'c'}, /*Total number of 2D projections used for reconstruction.*/
               {"vox_wid",  required_argument, 0, 'd'}, /*Side length of a cubic voxel in inverse units of linear attenuation coefficient of the object. 
		For example, if units of "vox_wid" is mm, then attenuation coefficient will have units of mm^-1, and vice versa.
		Note that attenuation coefficient is what we are trying to reconstruct.*/
               {"rot_center",    required_argument, 0, 'e'}, /*Center of rotation of object, in units of detector pixels. 
		For example, if center of rotation is exactly at the center of the object, then rot_center = proj_num_cols/2.
		If not, then specify as to which detector column does the center of rotation of the object projects to. */
               {"mag_sigma",  required_argument, 0, 'f'}, /*Spatial regularization parameter of the prior model.*/
               {"mag_c",  required_argument, 0, 'g'}, 
		/*parameter of the spatial qGGMRF prior model. 
 		Should be fixed to be much lesser (typically 0.01 times) than the ratio of voxel difference over an edge to sigma_s.*/ 
               {"elec_sigma",  required_argument, 0, 'h'}, /*Spatial regularization parameter of the prior model.*/
               {"elec_c",  required_argument, 0, 'i'}, 
		/*parameter of the spatial qGGMRF prior model. 
 		Should be fixed to be much lesser (typically 0.01 times) than the ratio of voxel difference over an edge to sigma_s.*/ 
               {"convg_thresh",    required_argument, 0, 'j'}, /*Used to determine when the algorithm is converged at each stage of multi-resolution.
		If the ratio of the average magnitude of voxel updates to the average voxel value expressed as a percentage is less
		than "convg_thresh" then the algorithm is assumed to have converged and the algorithm stops.*/
               {"restart",    no_argument, 0, 'k'}, /*If the reconstruction gets killed due to any unfortunate reason (like exceeding walltime in a super-computing cluster), use this flag to restart the reconstruction from the beginning of the current multi-resolution stage. Don't use restart if WRITE_EVERY_ITER  is 1.*/
		{0, 0, 0, 0}
         };

	*restart = 0;
	while(1)
	{		
	   c = getopt_long (argc, argv, "a:b:c:d:e:f:g:h:i:j:k", long_options, &option_index);
           /* Detect the end of the options. */
          if (c == -1) break;
	  switch (c) { 
		case  0 : fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Argument not recognized\n");		break;
		case 'a': *proj_rows = (int32_t)atoi(optarg);			break;
		case 'b': *proj_cols = (int32_t)atoi(optarg);			break;
		case 'c': *proj_num = (int32_t)atoi(optarg);			break;
		case 'd': *vox_wid = (float)atof(optarg);			break;
		case 'e': *rot_center = (float)atof(optarg);			break;
		case 'f': *mag_sigma = (float)atof(optarg);			break;
		case 'g': *mag_c = (float)atof(optarg);				break;
		case 'h': *elec_sigma = (float)atof(optarg);			break;
		case 'i': *elec_c = (float)atof(optarg);				break;
		case 'j': *convg_thresh = (float)atof(optarg);			break;
		case 'k': *restart = 1;		break;
		case '?': fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Cannot recognize argument %s\n",optarg); break;
		}
	}

	if(argc-optind > 0)
		fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Argument list has an error\n");
}


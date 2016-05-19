#include <stdio.h>
#include <stdint.h>
#include <mpi.h>
#include <stdlib.h>
#include <getopt.h>
#include "XT_Main.h"
#include "pcmbir4d.h"

/*Function prototype definitions which will be defined later in the file.*/
void read_data (float **measurements, float **brights, float **proj_angles, float **proj_times, float **recon_times, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, int32_t recon_num, float vox_wid, float rot_center, float obj2det_dist, float light_energy, float pag_regparam, uint8_t recon_type, FILE* debug_file_ptr);
void read_command_line_args (int32_t argc, char **argv, int32_t *proj_rows, int32_t *proj_cols, int32_t *proj_num, int32_t *recon_num, float *vox_wid, float *rot_center, float *mag_sig_s, float *mag_sig_t, float *mag_c_s, float *mag_c_t, float *phase_sig_s, float *phase_sig_t, float *phase_c_s, float *phase_c_t, float *convg_thresh, float *obj2det_dist, float *xray_energy, float *pag_regparam, uint8_t *restart, uint8_t *recon_type, FILE* debug_msg_ptr);

/*The main function which reads the command line arguments, reads the data,
  and does the reconstruction.*/
int main(int argc, char **argv)
{
	uint8_t restart, recon_type;
	int32_t proj_rows, proj_cols, proj_num, recon_num, nodes_num, nodes_rank;
	float *magobject, *phaseobject, *measurements, *brights, *proj_angles, *proj_times, *recon_times, vox_wid, rot_center, mag_sig_s, mag_sig_t, mag_c_s, mag_c_t, phase_sig_s, phase_sig_t, phase_c_s, phase_c_t, convg_thresh, obj2det_dist, xray_energy, pag_regparam;
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
	read_command_line_args (argc, argv, &proj_rows, &proj_cols, &proj_num, &recon_num, &vox_wid, &rot_center, &mag_sig_s, &mag_sig_t, &mag_c_s, &mag_c_t, &phase_sig_s, &phase_sig_t, &phase_c_s, &phase_c_t, &convg_thresh, &obj2det_dist, &xray_energy, &pag_regparam, &restart, &recon_type, debug_msg_ptr);
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Number of nodes is %d and command line input argument values are proj_rows = %d, proj_cols = %d, proj_num = %d, recon_num = %d, vox_wid = %e, rot_center = %e, mag_sig_s = %e, mag_sig_t = %e, mag_c_s = %e, mag_c_t = %e, phase_sig_s = %e, phase_sig_t = %e, phase_c_s = %e, phase_c_t = %e, convg_thresh = %e, obj2det_dist = %f, xray_energy = %f, pag_regparam = %f, restart = %d, recon_type = %d\n", nodes_num, proj_rows, proj_cols, proj_num, recon_num, vox_wid, rot_center, mag_sig_s, mag_sig_t, mag_c_s, mag_c_t, phase_sig_s, phase_sig_t, phase_c_s, phase_c_t, convg_thresh, obj2det_dist, xray_energy, pag_regparam, restart, recon_type);	
	
	/*Allocate memory for data arrays used for reconstruction.*/
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Allocating memory for data ....\n");

	/*Read data*/
	if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Reading data ....\n");
	read_data (&measurements, &brights, &proj_angles, &proj_times, &recon_times, proj_rows, proj_cols, proj_num, recon_num, vox_wid, rot_center, obj2det_dist, xray_energy, pag_regparam, recon_type, debug_msg_ptr);
	
	if (recon_type > 0)
	{
		if (nodes_rank == 0) fprintf(debug_msg_ptr, "main: Reconstructing the data ....\n");
		/*Run the reconstruction*/
		phcontomo_reconstruct (&magobject, &phaseobject, measurements, brights, proj_angles, proj_times, recon_times, proj_rows, proj_cols, proj_num, recon_num, vox_wid, rot_center, mag_sig_s, mag_sig_t, mag_c_s, mag_c_t, phase_sig_s, phase_sig_t, phase_c_s, phase_c_t, convg_thresh, obj2det_dist, xray_energy, pag_regparam, restart, recon_type, debug_msg_ptr);
		free(magobject);
		free(phaseobject);
	}
	
	free(measurements);
	free(brights);
	free(proj_angles);
	free(proj_times);
	free(recon_times);

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


void read_data (float **measurements, float **brights, float **proj_angles, float **proj_times, float **recon_times, int32_t proj_rows, int32_t proj_cols, int32_t proj_num, int32_t recon_num, float vox_wid, float rot_center, float obj2det_dist, float light_energy, float pag_regparam, uint8_t recon_type, FILE* debug_file_ptr)
{
	char measurements_filename[] = MEASUREMENTS_FILENAME;
	char brights_filename[] = BRIGHTS_FILENAME;
	char proj_angles_filename[] = PROJ_ANGLES_FILENAME;
	char proj_times_filename[] = PROJ_TIMES_FILENAME;
	char recon_times_filename[] = RECON_TIMES_FILENAME;
	int32_t i, offset, size, rank, num_nodes;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
	*proj_angles = (float*)calloc (proj_num, sizeof(float));
	*proj_times = (float*)calloc (proj_num, sizeof(float));
	*recon_times = (float*)calloc (recon_num + 1, sizeof(float));
	
/*	read_BinFile (proj_angles_filename, *proj_angles, 0, proj_num, debug_file_ptr);
	read_BinFile (proj_times_filename, *proj_times, 0, proj_num, debug_file_ptr);
	read_BinFile (recon_times_filename, *recon_times, 0, recon_num + 1, debug_file_ptr);*/

	for (i = 0; i < proj_num; i++)
	{
/*		(*proj_angles)[i] = M_PI*((float)i)/proj_num;*/
/*HACK HACK*/		(*proj_angles)[i] = M_PI/2; /*HACK HACK*/
		(*proj_times)[i] = i;
	}
	
	(*recon_times)[0] = 0;
	(*recon_times)[1] = proj_num ;

	*measurements = (float*)calloc ((proj_num*proj_rows*proj_cols)/num_nodes, sizeof(float));
	*brights = (float*)calloc ((proj_rows*proj_cols)/num_nodes, sizeof(float));
	if (recon_type == 0)
	{
		phcontomo_forward_project (measurements, brights, *proj_angles, proj_rows, proj_cols, proj_num, vox_wid, rot_center, obj2det_dist, light_energy, pag_regparam, debug_file_ptr);
		size = proj_rows*proj_cols/num_nodes;
		for (i = 0; i < proj_num; i++)
		{
			offset = i*proj_rows*proj_cols + rank*size;
			write_BinFile (measurements_filename, *measurements + i*size, offset, size, debug_file_ptr);
		}
		write_BinFile (brights_filename, *brights, rank*size, size, debug_file_ptr);
	}
	else 
	{
		size = proj_rows*proj_cols/num_nodes;
		for (i = 0; i < proj_num; i++)
		{
			offset = i*proj_rows*proj_cols + rank*size;
			read_BinFile (measurements_filename, *measurements + i*size, offset, size, debug_file_ptr);
		}
		read_BinFile (brights_filename, *brights, rank*size, size, debug_file_ptr);
	}
}

void read_command_line_args (int32_t argc, char **argv, int32_t *proj_rows, int32_t *proj_cols, int32_t *proj_num, int32_t *recon_num, float *vox_wid, float *rot_center, float *mag_sig_s, float *mag_sig_t, float *mag_c_s, float *mag_c_t, float *phase_sig_s, float *phase_sig_t, float *phase_c_s, float *phase_c_t, float *convg_thresh, float *obj2det_dist, float *xray_energy, float *pag_regparam, uint8_t *restart, uint8_t *recon_type, FILE* debug_msg_ptr)
/*Function which parses the command line input to the C code and initializes several variables.*/
{
	int32_t option_index;
	char c;
	static struct option long_options[] =
        {
               {"proj_rows",  required_argument, 0, 'a'}, /*Number of columns in the projection image. Typically, it is the number of detector bins in the cross-axial direction.*/
               {"proj_cols",  required_argument, 0, 'b'}, /*Number of rows (or slices) in the projection image. Typically, it is the number of detector bins in the axial direction.*/
               {"proj_num",  required_argument, 0, 'c'}, /*Total number of 2D projections used for reconstruction.*/
               {"recon_num",  required_argument, 0, 'd'}, /*Number of 3D time samples in the 4D reconstruction. For 3D reconstructions, this value should be set to 1.*/
               {"vox_wid",  required_argument, 0, 'e'}, /*Side length of a cubic voxel in inverse units of linear attenuation coefficient of the object. 
		For example, if units of "vox_wid" is mm, then attenuation coefficient will have units of mm^-1, and vice versa.
		Note that attenuation coefficient is what we are trying to reconstruct.*/
               {"rot_center",    required_argument, 0, 'f'}, /*Center of rotation of object, in units of detector pixels. 
		For example, if center of rotation is exactly at the center of the object, then rot_center = proj_num_cols/2.
		If not, then specify as to which detector column does the center of rotation of the object projects to. */
               {"mag_sig_s",  required_argument, 0, 'g'}, /*Spatial regularization parameter of the prior model.*/
               {"mag_sig_t",  required_argument, 0, 'h'}, /*Temporal regularization parameter of the prior model.*/
               {"mag_c_s",  required_argument, 0, 'i'}, 
		/*parameter of the spatial qGGMRF prior model. 
 		Should be fixed to be much lesser (typically 0.01 times) than the ratio of voxel difference over an edge to sigma_s.*/ 
               {"mag_c_t",  required_argument, 0, 'j'}, 
		/*parameter of the temporal qGGMRF prior model. 
  		Should be fixed to be much lesser (typically 0.01 times) than the ratio of voxel difference over an edge to sigma_t.*/ 
               {"convg_thresh",    required_argument, 0, 'k'}, /*Used to determine when the algorithm is converged at each stage of multi-resolution.
		If the ratio of the average magnitude of voxel updates to the average voxel value expressed as a percentage is less
		than "convg_thresh" then the algorithm is assumed to have converged and the algorithm stops.*/
               {"restart",    no_argument, 0, 'n'}, /*If the reconstruction gets killed due to any unfortunate reason (like exceeding walltime in a super-computing cluster), use this flag to restart the reconstruction from the beginning of the current multi-resolution stage. Don't use restart if WRITE_EVERY_ITER  is 1.*/
               {"phase_sig_s",  required_argument, 0, 'o'}, /*Spatial regularization parameter of the prior model.*/
               {"phase_sig_t",  required_argument, 0, 'p'}, /*Temporal regularization parameter of the prior model.*/
               {"phase_c_s",  required_argument, 0, 'q'}, 
               {"phase_c_t",  required_argument, 0, 'r'}, 
               {"gen_data",    no_argument, 0, 's'}, 
               {"pag_mbir",    no_argument, 0, 't'}, 
               {"critir",    no_argument, 0, 'u'}, 
               {"obj2det_dist",    required_argument, 0, 'v'}, 
               {"xray_energy",    required_argument, 0, 'w'}, 
               {"pag_regparam",    required_argument, 0, 'x'}, 
		{0, 0, 0, 0}
         };

	*recon_type = -1;
	*restart = 0;
	while(1)
	{		
	   c = getopt_long (argc, argv, "a:b:c:d:e:f:g:h:i:j:k:no:p:q:r:stuv:w:x:", long_options, &option_index);
           /* Detect the end of the options. */
          if (c == -1) break;
	  switch (c) { 
		case  0 : fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Argument not recognized\n");		break;
		case 'a': *proj_rows = (int32_t)atoi(optarg);			break;
		case 'b': *proj_cols = (int32_t)atoi(optarg);			break;
		case 'c': *proj_num = (int32_t)atoi(optarg);			break;
		case 'd': *recon_num = (int32_t)atoi(optarg);			break;
		case 'e': *vox_wid = (float)atof(optarg);			break;
		case 'f': *rot_center = (float)atof(optarg);			break;
		case 'g': *mag_sig_s = (float)atof(optarg);			break;
		case 'h': *mag_sig_t = (float)atof(optarg);			break;
		case 'i': *mag_c_s = (float)atof(optarg);				break;
		case 'j': *mag_c_t = (float)atof(optarg);				break;
		case 'k': *convg_thresh = (float)atof(optarg);			break;
		case 'n': *restart = 1;		break;
		case 'o': *phase_sig_s = (float)atof(optarg);			break;
		case 'p': *phase_sig_t = (float)atof(optarg);			break;
		case 'q': *phase_c_s = (float)atof(optarg);				break;
		case 'r': *phase_c_t = (float)atof(optarg);				break;
		case 's': *recon_type = 0;					break;
		case 't': *recon_type = 1;					break;
		case 'u': *recon_type = 2;					break;
		case 'v': *obj2det_dist = (float)atof(optarg);			break;
		case 'w': *xray_energy = (float)atof(optarg);			break;
		case 'x': *pag_regparam = (float)atof(optarg);			break;
		case '?': fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Cannot recognize argument %s\n",optarg); break;
		}
	}

	if(argc-optind > 0)
		fprintf(debug_msg_ptr, "ERROR: read_command_line_args: Argument list has an error\n");
}


#mpiexec -n 1 ./XT_Main --proj_rows 16 --proj_cols 768 --proj_num 6144 --recon_num 128 --vox_wid 1.3 --rot_center 385 --sig_s 0.010413 --sig_t 0.000104 --c_s 0.000001 --c_t 0.000001 --convg_thresh 1 --remove_rings 1 --remove_streaks 1  

cd $PBS_O_WORKDIR
module load devel
module load valgrind
export PARALLEL=1
export OMP_NUM_THREADS=32

./XT_Main --proj_rows 256 --proj_cols 256 --proj_x_num 101 --proj_y_num 101 --vox_wid 6 --qggmrf_sigma 0.00001 --qggmrf_c 0.000001 --convg_thresh 0.01 --admm_mu 100000000000 --admm_maxiters 20 --x_widnum 512 --y_widnum 512 --z_widnum 64 --data_variance 0.000146 

checkjob -v $PBS_JOBID

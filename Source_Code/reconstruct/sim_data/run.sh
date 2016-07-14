#mpiexec -n 1 ./XT_Main --proj_rows 16 --proj_cols 768 --proj_num 6144 --recon_num 128 --vox_wid 1.3 --rot_center 385 --sig_s 0.010413 --sig_t 0.000104 --c_s 0.000001 --c_t 0.000001 --convg_thresh 1 --remove_rings 1 --remove_streaks 1  

cd $PBS_O_WORKDIR
module load devel
module load valgrind
export PARALLEL=1
export OMP_NUM_THREADS=32

# Arguments for simulated data generated by MBIR forward projector
./XT_Main --proj_rows 256 --proj_cols 256 --proj_num 71 --vox_wid 5 --rot_center 128 --mag_sigma 0.0002 --mag_c 0.000001 --elec_sigma 1 --elec_c 0.000001 --convg_thresh 0.1 --admm_mu 1000000

# mpiexec -n 1 -machinefile nodefile valgrind --tool=memcheck --leak-check=full --show-reachable=yes --track-origins=yes --error-limit=no ./XT_Main --proj_rows 64 --proj_cols 64 --proj_num 71 --vox_wid 2 --rot_center 32 --mag_sigma 0.1 --mag_c 0.000001 --elec_sigma 0.0001 --elec_c 0.000001 --convg_thresh 5


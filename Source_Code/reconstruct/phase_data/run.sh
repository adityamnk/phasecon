#mpiexec -n 1 ./XT_Main --proj_rows 16 --proj_cols 768 --proj_num 6144 --recon_num 128 --vox_wid 1.3 --rot_center 385 --sig_s 0.010413 --sig_t 0.000104 --c_s 0.000001 --c_t 0.000001 --convg_thresh 1 --remove_rings 1 --remove_streaks 1  

#cd $PBS_O_WORKDIR
module load devel
export PARALLEL=1
export OMP_NUM_THREADS=32
uniq < $PBS_NODEFILE > nodefile

#mpiexec -n 1 -machinefile nodefile ./XT_Main --proj_rows 64 --proj_cols 64 --proj_num 64 --recon_num 1 --vox_wid 0.1 --rot_center 32 --mag_sig_s 0.000005 --mag_sig_t 1 --mag_c_s 0.000001 --mag_c_t 0.000001 --phase_sig_s 0.0000015 --phase_sig_t 1 --phase_c_s 0.000001 --phase_c_t 0.000001 --convg_thresh 1 --obj2det_dist 20000 --xray_energy 10 --pag_regparam 90 --gen_data > debug_data.log
#mpiexec -n 1 -machinefile nodefile ./XT_Main --proj_rows 64 --proj_cols 64 --proj_num 64 --recon_num 1 --vox_wid 0.1 --rot_center 32 --mag_sig_s 0.0000005 --mag_sig_t 1 --mag_c_s 0.000001 --mag_c_t 0.000001 --phase_sig_s 0.0000001 --phase_sig_t 1 --phase_c_s 0.000001 --phase_c_t 0.000001 --convg_thresh 1 --obj2det_dist 20000 --xray_energy 10 --pag_regparam 90 --pag_mbir > debug_pag.log 
#mpiexec -n 1 -machinefile nodefile ./XT_Main --proj_rows 64 --proj_cols 64 --proj_num 64 --recon_num 1 --vox_wid 0.1 --rot_center 32 --mag_sig_s 0.000005 --mag_sig_t 1 --mag_c_s 0.000001 --mag_c_t 0.000001 --phase_sig_s 0.00005 --phase_sig_t 1 --phase_c_s 0.000001 --phase_c_t 0.000001 --convg_thresh 1 --obj2det_dist 20000 --xray_energy 10 --pag_regparam 90 --critir > debug_critir.log 

#mpiexec -n 1 -machinefile nodefile ./XT_Main --proj_rows 32 --proj_cols 128 --proj_num 128 --recon_num 1 --vox_wid 0.5 --rot_center 64 --mag_sig_s 0.000005 --mag_sig_t 1 --mag_c_s 0.000001 --mag_c_t 0.000001 --phase_sig_s 0.0000015 --phase_sig_t 1 --phase_c_s 0.000001 --phase_c_t 0.000001 --convg_thresh 1 --obj2det_dist 100000 --xray_energy 3 --pag_regparam 97.8734 --gen_data > debug_data.log
#mpiexec -n 1 -machinefile nodefile ./XT_Main --proj_rows 32 --proj_cols 128 --proj_num 128 --recon_num 1 --vox_wid 0.5 --rot_center 64 --mag_sig_s 0.0005 --mag_sig_t 1 --mag_c_s 0.000001 --mag_c_t 0.000001 --phase_sig_s 0.0001 --phase_sig_t 1 --phase_c_s 0.000001 --phase_c_t 0.000001 --convg_thresh 1 --obj2det_dist 100000 --xray_energy 3 --pag_regparam 97.8734 --pag_mbir > debug_pag.log 
mpiexec -n 1 -machinefile nodefile ./XT_Main --proj_rows 32 --proj_cols 128 --proj_num 128 --recon_num 1 --vox_wid 0.5 --rot_center 64 --mag_sig_s 0.00005 --mag_sig_t 1 --mag_c_s 0.000001 --mag_c_t 0.000001 --phase_sig_s 0.0005 --phase_sig_t 1 --phase_c_s 0.000001 --phase_c_t 0.000001 --convg_thresh 1 --obj2det_dist 100000 --xray_energy 3 --pag_regparam 97.8734 --critir > debug_critir.log 

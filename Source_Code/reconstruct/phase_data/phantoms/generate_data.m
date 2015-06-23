clc; clear; close all;

NP_xyz = 256;
dwnsmpl = 2;
proj_angles = 0:180/(NP_xyz/dwnsmpl):180-180/(NP_xyz/dwnsmpl);
proj_times = 0:(NP_xyz/dwnsmpl)-1;
recon_times = [0,NP_xyz/dwnsmpl];
noise_sd = 1;

mag_filename = 'mag_phantom.bin';
phase_filename = 'phase_phantom.bin';

measurements_filename = 'measurements.bin';
weights_filename = 'weights.bin';
proj_angles_filename = 'proj_angles.bin';
proj_times_filename = 'proj_times.bin';
recon_times_filename = 'recon_times.bin';

[mag, phase] = read_phantom(mag_filename, phase_filename, NP_xyz, NP_xyz);
measurements = forward_project_phantom (mag, phase, proj_angles, noise_sd, dwnsmpl);
weights = ones(size(measurements));

write_data(measurements_filename, measurements);
write_data(weights_filename, weights);
write_data(proj_angles_filename, proj_angles*pi/180);
write_data(proj_times_filename, proj_times);
write_data(recon_times_filename, recon_times);

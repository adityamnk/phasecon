clc; clear; close all;

NP_xyz = 256;
dwnsmpl = 2;
theta = 0:180/(NP_xyz/dwnsmpl):180-180/(NP_xyz/dwnsmpl);

mag_filename = 'mag_phantom.bin';
phase_filename = 'phase_phantom.bin';
data_filename = 'measurements.bin';
noise_sd = 1;

[mag, phase] = read_phantom(mag_filename, phase_filename, NP_xyz, NP_xyz);
[data] = forward_project_phantom (mag, phase, theta, noise_sd, dwnsmpl);

write_data(data_filename, data);


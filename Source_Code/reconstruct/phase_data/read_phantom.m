function [mag,phase] = read_phantom(mag_filename, phase_filename, N_z, N_xy)

fidmag = fopen(mag_filename, 'r');
fidphase = fopen(phase_filename, 'r');

mag = fread(fidmag, N_z*N_xy*N_xy, 'float');
mag = reshape(mag,[N_xy,N_xy,N_z]);
phase = fread(fidphase, N_z*N_xy*N_xy, 'float');
phase = reshape(phase,[N_xy,N_xy,N_z]);

fclose(fidmag);
fclose(fidphase);
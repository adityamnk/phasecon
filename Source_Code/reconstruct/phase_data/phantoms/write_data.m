function [] = write_data(data_filename, data)

fid = fopen(data_filename, 'w');
fwrite(fid, data, 'float');
fclose(fid);
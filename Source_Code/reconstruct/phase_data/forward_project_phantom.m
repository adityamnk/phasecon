function [fftdata] = forward_project_phantom (mag, phase, theta, noise_sd, dwnsmpl)

sz = size(mag);
data = zeros(sz(1),sz(3),numel(theta));

for j = 1:numel(theta)
    for i = 1:sz(3)
        data_mag = radon(mag(:,:,i), theta(j));
        data_phase = radon(phase(:,:,i), theta(j));
        temp = exp(-(data_mag + i*data_phase));
        data(:,i,j) = temp(floor((numel(temp) - sz(1))/2) : floor((numel(temp)-sz(1))/2) + sz(1) - 1); 
    end
end

fftdata = zeros(sz(1)/dwnsmpl,sz(3)/dwnsmpl,numel(theta));
for j = 1:numel(theta)
    fftdata(:,:,j) = imresize(fft2(data(:,:,j)), 1.0/dwnsmpl);
end

fftdata = abs(fftdata) + noise_sd*norm(size(fftdata));



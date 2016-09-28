function [delta_, loss_] = L2Loss(active_, label)

[r,c,cn,bz] = size(active_);
delta_ = zeros(r,c,cn,bz);


dt = active_ - label;
loss_ = 0.5 * sum(dt(:).^2);
delta_ = dt/bz;
loss_ = loss_/bz;


end
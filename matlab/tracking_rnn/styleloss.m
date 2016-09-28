function [delta_, loss_] = styleloss(active_, label)

[r,c,cn,bz] = size(active_);
active_ = reshape(active_,[r*c,cn,bz]);
label = reshape(label,[r*c,cn,bz]);
N = cn;
M = r*c;
dt = zeros(size(label));
loss = 0;

for i=1:bz
    
   F = active_(:,:,i);
   G = F' * F;
   L = label(:,:,i);
   A = L' * L;
   
   diff = G -A;
   dt(:,:,i) = F * diff; 
   loss = loss + sum(diff(:).^2) / (4*N*N*M*M);
    
end
dt = reshape(dt,[r,c,cn,bz]);
delta_ = dt/bz;
loss_ = loss/bz;


end

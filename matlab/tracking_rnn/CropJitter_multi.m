function [simg, sbb] = CropJitter_multi(img, bb, STR)
% for multiple images (e.g. one batch)
% bbs: n*4; ch: n*3
[r,c,ch,~] = size(img);
[n,~] = size(bb);
px = [bb(:,1); bb(:,3); bb(:,5); bb(:,7)]; % 2n
py = [bb(:,2); bb(:,4); bb(:,6); bb(:,8)]; % 2n
sbb = bb;
while 1
    rate =  (rand - 0.5)/10;
    shift_x = floor(max(r,c) * rate);
    rate =  (rand - 0.5)/10;
    shift_y = floor(max(r,c) * rate);
    scale_x = 1+(rand-0.5)/10;
    scale_y =  scale_x;
    angle = (rand - 0.5)*(15/180)*pi;
%     angle = 0;
    A = [scale_x * cos(angle), scale_y * sin(angle), shift_x;...
        -scale_x * sin(angle), scale_y * cos(angle), shift_y]';
    T = maketform('affine', A);
    [pX, pY] = tformfwd(T, px, py);
    
    if (min(pX) > 4 && max(pX) < c-4)...
            &&(min(pY) > 4 && max(pY) < r-4)
        break;
    end
end
imgc = reshape(img,[r, c, ch * n]);
simgc = imtransform(imgc, T, 'XData',[1,c], 'YData',[1,r]);
img = reshape(simgc,[r, c, ch, n]);
% resize to 224
simg = zeros(STR.patchsize,STR.patchsize,ch,n);
for m = 1:n
    simg(:,:,:,m) = imresize(img(:,:,:,m),[STR.patchsize,STR.patchsize]);
    if rand > 0.6 && ch == 3
        simg(:,:,:,m) = im2uint8(imrandcolor(uint8(simg(:,:,:,m))));
    end
end
for k = 1:4 % 4 points
    sbb(:,(k-1)*2+1) = pX((k-1)*n+1:k*n)*(STR.patchsize/c);
    sbb(:,k*2) = pY((k-1)*n+1:k*n)*(STR.patchsize/r);
end

simg = single(simg);

end
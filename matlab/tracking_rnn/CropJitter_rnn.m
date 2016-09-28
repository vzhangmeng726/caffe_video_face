function [simg, sbb] = CropJitter_rnn(img, bb, STR)
[r,c,ch] = size(img);
if bb(1) < 1, bb(1) = 1;end
if bb(2) < 1, bb(2) = 1;end
if bb(1)+bb(3) > c, bb(3) = c-bb(1);end
if bb(2)+bb(4) > r, bb(4) = r-bb(2);end
px = [bb(1)+1; bb(1)+bb(3)];
py = [bb(2)+1; bb(2)+bb(4)];
current_iter = STR.Solver_.iter();
iter = 1;
while iter < 10     
    if current_iter < 500
        rng('shuffle');   
        rate =  (rand - 0.5)/10;
        shift_x = floor(max(r,c) * rate);
        shift_y = floor(max(r,c) * rate);
        scale_x = 1+(rand-0.5)/10;
        scale_y =  scale_x;
    else
        shift_x = 0;
        shift_y = 0;
        scale_x = 1;
        scale_y = 1;
    end    
    %     angle = (rand - 0.5)*(20/180)*pi;
    angle = 0;
    A = [scale_x * cos(angle), scale_y * sin(angle), shift_x;...
        -scale_x * sin(angle), scale_y * cos(angle), shift_y]';
    T = maketform('affine', A);
    img = imtransform(img, T, 'XData',[1,c], 'YData',[1,r]);
    [px, py] = tformfwd(T, px, py);
    
    if (px(1) > 1 && px(2) < c)&&(py(1) > 1 && py(2) < r)
        break;
    end
    iter = iter + 1;
end
% resize to 224
img = imresize(img,[STR.patchsize,STR.patchsize]);
sbb(1) = round(px(1)*(STR.patchsize/c));
sbb(2) = round(py(1)*(STR.patchsize/r));
sbb(3) = round((px(2)-px(1))*(STR.patchsize/c));
sbb(4) = round((py(2)-py(1))*(STR.patchsize/r));

if sbb(1) < 1, sbb(1) = 1;end
if sbb(2) < 1, sbb(2) = 1;end
if sbb(1)+sbb(3) > c, sbb(3) = c-sbb(1);end
if sbb(2)+sbb(4) > r, sbb(4) = r-sbb(2);end

if rand > 0.6 && ch == 3
    img = im2uint8(imrandcolor(uint8(img)));
end
simg = single(img);

end
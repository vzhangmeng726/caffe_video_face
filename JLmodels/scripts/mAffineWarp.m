function output = mAffineWarp(T_, im, parm)
[r,c,ch] = size(im);
output = zeros(parm.patchsize,parm.patchsize,ch);
    for y = 1:parm.patchsize
        for x = 1:parm.patchsize
            fx = (T_(1,1)*x + T_(1,2)*y + T_(1,3));
            fy = (T_(2,1)*x + T_(2,2)*y + T_(2,3));
            sx = floor(fx); sy = floor(fy);
            
            fx = fx - sx; fy = fy - sy;
            
            sy = max(1, min(sy, r-1));
            sx = max(1, min(sx, c-1));
            
            w_y0 = abs(1 - fy); w_y1 = abs(fy);
            w_x0 = abs(1 - fx); w_x1 = abs(fx);
            
            for k = 1:ch
               output(y,x,k) = im(sy,sx,k) * w_x0 * w_y0 + ...
                   im(sy+1,sx,k) * w_x0 * w_y1 + ...
                   im(sy,sx+1,k) * w_x1 * w_y0 + ...
                   im(sy+1, sx+1, k) * w_x1 * w_y1;
            end
        end
    end
end
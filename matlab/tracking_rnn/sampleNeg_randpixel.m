function mask = sampleNeg_randpixel(bb, gt)
% debug version
rate = 0.1;
[r,c,~] = size(gt);
posnum = bb(3) * bb(4);
% negnum = posnum * 10;
% mask = rand(r,c) < (negnum/(r*c));
mask = rand(r,c) < 0.5;
ebb(1) = bb(1) - round(rate*bb(3));
ebb(2) = bb(2) - round(rate*bb(4));
ebb(3) = bb(1) + bb(3) + round(rate*bb(3));
ebb(4) = bb(2) + bb(4) + round(rate*bb(4));
if ebb(1) < 1, ebb(1)=1; end
if ebb(2) < 1, ebb(2)=1; end
if ebb(3) > c, ebb(3) = c; end
if ebb(4) > r, ebb(4) = r; end
mask(ebb(2)+1:ebb(4),ebb(1)+1:ebb(3)) = 0;
mask(bb(2)+1:bb(2)+bb(4), bb(1)+1:bb(1)+bb(3)) = 1;
end
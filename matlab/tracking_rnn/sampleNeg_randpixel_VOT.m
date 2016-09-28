function mask = sampleNeg_randpixel_VOT(bb, gt)
% debug version
rate = 0.1;
[r,c,~] = size(gt);
posnum = sum(gt(:));
mask = rand(r,c) < 0.5;
width = norm(bb([1,2])-bb([3,4]));
height = norm(bb([3,4])-bb([5,6]));
bx = round(rate * (width + height));
se = strel('square', bx);
gt_tmp = imdilate(gt,se);
mask(gt_tmp > 0) = 0;
mask(gt > 0) = 1;
end
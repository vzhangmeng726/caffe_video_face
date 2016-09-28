function mask = sampleNeg_hardpixel_st2_VOT(bb, dt, gt)
% debug version
rate = 0.5;
[r,c,~] = size(gt);
posnum = sum(gt(:));
negnum = posnum * 10;
if negnum > r * c
    negnum = r*c - posnum;
end
mask = 0.1 * ones(r,c) + single(rand(r,c) < (negnum/(r*c)));
[y,ind] = sort(abs(dt(:)),'descend');
if length(find(abs(y) > 1e-3)) < negnum
    mask(abs(dt) > 1e-3) = mask(abs(dt) > 1e-3)+1;
else
    mask(ind(1:negnum)) = mask(ind(1:negnum))+1;
end
width = norm(bb([1,2])-bb([3,4]));
height = norm(bb([3,4])-bb([5,6]));
bx = round(rate * (width + height));
se = strel('square', bx);
gt_tmp = imdilate(gt,se);
mask(gt_tmp > 0) = 3;
end
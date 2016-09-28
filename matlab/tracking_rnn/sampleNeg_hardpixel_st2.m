function mask = sampleNeg_hardpixel_st2(bb, dt, gt)
% debug version
rate = 0.5;
[r,c,~] = size(gt);
posnum = bb(3) * bb(4);
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
ebb(1) = bb(1) - round(rate*bb(3));
ebb(2) = bb(2) - round(rate*bb(4));
ebb(3) = bb(1) + bb(3) + round(rate*bb(3));
ebb(4) = bb(2) + bb(4) + round(rate*bb(4));
if ebb(1) < 1, ebb(1)=1; end
if ebb(2) < 1, ebb(2)=1; end
if ebb(3) > c, ebb(3) = c; end
if ebb(4) > r, ebb(4) = r; end
mask(ebb(2)+1:ebb(4),ebb(1)+1:ebb(3)) = 3;
% mask(bb(2)+1:bb(2)+bb(4), bb(1)+1:bb(1)+bb(3)) =...
%     mask(bb(2)+1:bb(2)+bb(4), bb(1)+1:bb(1)+bb(3)) + 3;
end
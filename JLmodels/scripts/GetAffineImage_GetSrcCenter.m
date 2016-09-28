function [x, y] = GetAffineImage_GetSrcCenter(landmark)
    len = length(landmark)/2;
    x = sum(landmark(1:2:end))/len;
    y = sum(landmark(2:2:end))/len;
end
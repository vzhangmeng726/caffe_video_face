function T_ = Get_Affine_matrix(src_center, dst_center, ang, scale)
% src/dst_center: vectors of x and y ;
T_ = zeros(2,3);
T_(1,1) = scale * cos(ang);
T_(1,2) = scale * sin(ang);
T_(2,1) = -T_(1,2);
T_(2,2) = T_(1,1);

T_(1,3) = src_center(1) - T_(1,1)*dst_center(1) - T_(1,2) * dst_center(2);
T_(2,3) = src_center(2) - T_(2,1)*dst_center(1) - T_(2,2) * dst_center(2);
end
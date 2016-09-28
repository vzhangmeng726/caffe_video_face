function [T,T_] = GetFaceAffineForm(parm, landmark)
[src_center(1),src_center(2)] = GetAffineImage_GetSrcCenter(landmark);
dst_center(1) = parm.patchsize/2; dst_center(2) = parm.patchsize/2;
scale = GetAffineImage_GetScale(parm, landmark);
ang = GetAffineImage_GetAngle(landmark);
T_ = Get_Affine_matrix(src_center, dst_center, -ang, scale);
T = maketform('affine',[T_; 0,0,1]');
end
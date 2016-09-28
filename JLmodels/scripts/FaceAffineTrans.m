function trans_im = FaceAffineTrans(landmark, im)
parm.patchsize = 128;
parm.norm_ratio = 0.3;
parm.mode = 'RECT_LE_RE_LM_RM';
[~,M] = GetFaceAffineForm(parm, landmark);
if size(im,3)==1
    im = repmat(im,[1,1,3]);
end
% % resize input
% im = imresize(im, [parm.patchsize,parm.patchsize]);
% tform
trans_im = uint8(mAffineWarp(M, im, parm));
end
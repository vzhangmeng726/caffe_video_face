function [delta_,loss_,iou_] = SigmSampledLoss_Single(active_, bbox, STR, mode)
current_iter = STR.Solver_.iter();
[r,c,cn,bz] = size(active_); % 85 channels
active_ = permute(active_,[2,1,3,4]);

act_ = active_(:,:,STR.video_id,:);
act_ = 1/(1+exp(-act_)); % sigmoid
act_ = squeeze(act_);

delta_ = zeros(r,c,cn,bz);
if bz~=STR.batchsize
    error('incorrect batch dimention of active_.');
end
loss = zeros(STR.batchsize,1);
iou = zeros(STR.batchsize,1);
rate = 0.1;
for m = 1:bz
    bb = bbox(m,:);
    % scale adjustion
    if r ~= STR.patchsize
        bb(1:2:end) = round(bb(1:2:end) * (c/STR.patchsize));
        bb(2:2:end) = round(bb(2:2:end) * (r/STR.patchsize));
    end
    gt = zeros(r,c);
    gt = roipoly(gt, bb(1:2:end), bb(2:2:end));
    
    try dt = act_(:,:,m) - gt;
    catch ME
        fprintf('size act: %d, gt: %d\n', size(act_(:,:,m)), size(gt));
        fprintf('bb: %d\n',bb);
        rethrow(ME)
    end
    loss(m) = 0.5 * sum(dt(:).^2);
    iou(m) = sum(sum((act_(:,:,m) > 0.5 & gt)))/...
        sum(sum((act_(:,:,m) > 0.5 | gt)));
    if isnan(iou(m))
        iou(m) = 0;
    end
    
    % option: determing the stage controled by "current_iter"
    if current_iter < 500
        mask = sampleNeg_randpixel_VOT(bb, gt);
    elseif current_iter < 3000
        mask = sampleNeg_hardpixel_VOT(bb, dt, gt);
    else
        mask = sampleNeg_hardpixel_st2_VOT(bb, dt, gt);
    end
    %
    if strcmp(mode,'pretrain')
        delta_(:,:,STR.video_id,m) = dt .* mask;
    end
end
delta_ = permute(delta_,[2,1,3,4]);
loss_ = mean(loss);
iou_ = mean(iou);
end
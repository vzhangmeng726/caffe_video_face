function [batch, bbox, STR] = GenerateSample_rnn(STR, mode)
% design: each batch contains the same video so that the batch images have
% the same size
% images in one batch are disordered
current_iter = STR.Solver_.iter();
batch = zeros(STR.patchsize, STR.patchsize, 3, STR.batchsize);
bbox = zeros(STR.batchsize,8);
rng('shuffle');
if current_iter < 1000
    sele_vd_ind = round(rand * length(STR.track_list));
    if sele_vd_ind < 1
        sele_vd_ind = 1;
    end
else
    if mod(current_iter, 1000) == 0 || ~isfield(STR,'distr')
        [~, STR.distr] = video_sorting(STR);
    end
    ix = randi(length(STR.distr),1);
    sele_vd_ind = STR.distr(ix);
end
STR.video_id = sele_vd_ind;


frame_folder = STR.frame_list{sele_vd_ind}; % obtain its frame folder
frame_dir = dir(fullfile(frame_folder,'*.jpg'));
frame_num = STR.num_list{sele_vd_ind};
if length(frame_dir) < frame_num
    error('please check frame: %s, frame_num: %d',frame_folder, frame_num);
end
rect_list = double(STR.rect_list{sele_vd_ind});
ind = round(rand *(frame_num-STR.batchsize));
if strcmp(mode,'pretrain')
    sind = ind+1:ind+STR.batchsize;
elseif strcmp(mode, 'validate') % validate with last ten frames
    sind = frame_num-STR.batchsize+1:frame_num;
end

for count = 1:STR.batchsize
    try 
        img = imread(fullfile(frame_folder, frame_dir(sind(count)).name));
    catch ME
        fprintf('frame: %s, sind: %d, count: %d',...
            frame_folder,sind(1),count);
        rethrow(ME)
    end
    if size(img,3) == 1
        img = repmat(img,[1,1,3]);
    end
    if count ==1 
        imgs = img;
    else
        imgs(:,:,:,count)=img;
    end  
end
bb = rect_list(sind,:);
[simg, sbb] = CropJitter_multi(imgs, bb, STR);
batch(:,:,:,:) = simg - repmat(STR.image_mean,[1,1,1,STR.batchsize]);
bbox(:,:) = sbb;
batch = BatchProcess(batch);
end
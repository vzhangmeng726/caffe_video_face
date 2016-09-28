function STR = Data_config(STR)


STR.patchsize = 224;
STR.batchsize = 20;
STR.posnumFrame = 50;
load('VGG_mean.mat');
STR.image_mean = image_mean;

STR.dataset = '/data/vot/vot2014';
STR.seqlist = fullfile(STR.dataset, 'list.txt');
% STR.dataset = '../../../DATA/VIDEO/tracking/VOT/';
fout = fopen(STR.seqlist);
a = textscan(fout, '%s');
fclose(fout);
track_list = a{1}; clear a
count = 1;

for m = 1:length(track_list)
    video_folder = fullfile(STR.dataset,track_list{m});
    % bbox
    bbox_file = fullfile(video_folder,'groundtruth.txt');
    fout = fopen(bbox_file);
    rects = textscan(fout, '%f,%f,%f,%f,%f,%f,%f,%f');
    rects = cell2mat(rects);
    num = size(rects,1);
    fclose(fout);
    % camera motion
    cm_file = fullfile(video_folder, 'camera_motion.label');
    fout = fopen(cm_file);
    cms = textscan(fout, '%d');
    cms = cell2mat(cms);
    fclose(fout);
    if length(cms)~=num
        error('check the number of camera motion labels.');
    end
    % illum_change
    illum_file = fullfile(video_folder, 'illum_change.label');
    fout = fopen(illum_file);
    illums = textscan(fout, '%d');
    illums = cell2mat(illums);
    fclose(fout);
    if length(illums)~=num
        error('check the number of illum_change labels.');
    end
    % motion_change
    motion_file = fullfile(video_folder, 'motion_change.label');
    fout = fopen(motion_file);
    motions = textscan(fout, '%d');
    motions = cell2mat(motions);
    fclose(fout);
    if length(motions)~=num
        error('check the number of motion_change labels.');
    end
    % occlusion
    occlusion_file = fullfile(video_folder, 'occlusion.label');
    fout = fopen(occlusion_file);
    occlusions = textscan(fout, '%d');
    occlusions = cell2mat(occlusions);
    fclose(fout);
    if length(occlusions)~=num
        error('check the number of occlusion labels.');
    end
    % size_change
    size_file = fullfile(video_folder, 'size_change.label');
    fout = fopen(size_file);
    sizes = textscan(fout, '%d');
    sizes = cell2mat(sizes);
    fclose(fout);
    if length(sizes)~=num
        error('check the number of occlusion labels.');
    end
    
    % image data
%     imlist = dir([video_folder,'/*.jpg']);
%     if length(imlist)~=num
%         error('check the number of image numbers.');
%     end
%     img=imread([video_folder,'/',imlist(1).name]);
%     imh=size(img,1);
%     imw=size(img,2);
%     ch=size(img,3);
%     if ch ~=3
%         error('assume image should be 3 channels.');
%     end
%     imgs=zeros(imh,imw,ch,num);
%     for imnum=1:num
%         img=imread([video_folder,'/',imlist(imnum).name]);
%         imgs(:,:,:,imnum)=img;
%     end
%     imgs=uint8(imgs);
    
    
    % store properties to STR
    STR.track_list{count} = track_list{m};
    STR.frame_list{count} = video_folder;
    STR.num_list{count} = num;
    STR.rect_list{count} = rects;
    STR.cm_list{count} = cms;
    STR.illum_list{count} = illums;
    STR.motion_list{count} = motions;
    STR.occlusion_list{count} = occlusions;
    STR.size_list{count} = sizes;
    %STR.image_list{count} = imgs;
    fprintf('finish load video %d ...\n',count);
    count = count + 1;
    
end

end
clear all;clc;
%% Setting
LFW_dir = '/home/wangzd/lfw-hallucination';
out_dir = '/home/wangzd/lfw-hallucination-aligned';
landmark_txt = '../lfw_testpairs_5pts.txt';

%% Processing

ID_list = dir(LFW_dir);
ID_list = {ID_list.name};
ID_list = ID_list(3:end);

landmark_list = importdata(landmark_txt,' ');
l_2 = landmark_list.data;
l_1 = landmark_list.textdata(:,3:12);
name_1 = landmark_list.textdata(:,2);
name_2 = landmark_list.textdata(:,13);
same = landmark_list.textdata(:,1);

for i=1:size(name_2,1)
    tmp = strsplit(name_2{i},'/');
    name_i = tmp{1};
    dir_name = fullfile(out_dir,name_i);
    if ~exist(dir_name)
        mkdir(dir_name);
    end
    img = imread(fullfile(LFW_dir,name_2{i}));
    img_ = FaceAffineTrans(l_2(i,:),img);
    
    imwrite(img_,fullfile(dir_name,tmp{2}))

end



for i=1:size(name_1,1)
    tmp = strsplit(name_1{i},'/');
    name_i = tmp{1};
    dir_name = fullfile(out_dir,name_i);
    if ~exist(dir_name)
        mkdir(dir_name);
    end
    img = imread(fullfile(LFW_dir,name_1{i}));
    img_ = FaceAffineTrans([eval(l_1{i,1}) eval(l_1{i,2}) eval(l_1{i,3}) eval(l_1{i,4})...
        eval(l_1{i,5}) eval(l_1{i,6}) eval(l_1{i,7}) eval(l_1{i,8}) eval(l_1{i,9}) eval(l_1{i,10})],img);
    imwrite(img_,fullfile(dir_name,tmp{2}))
end




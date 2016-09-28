
%%
clear all 
close all

warning off
mex_path = '../+caffe';
addpath(mex_path);
addpath('..');
addpath('../..');
addpath('../util');

%% Init
solver_file = './prototxt_vgg/solver.prototxt';
Solver = SolverParser(solver_file);
Solver.solver_.net.copy_from('./model_vgg/vgg16_onlyconv.caffemodel');
caffe.set_mode_gpu();


%% get data mean
sz=224;
datamean=zeros(sz,sz,3);
datamean(:,:,1)=123.68;
datamean(:,:,2)=116.779;
datamean(:,:,3)=103.939;

%% get style and context image
styleimg = double(imresize(imread('starry_night.jpg'),[sz,sz])) - datamean;
contextimg = double(imresize(imread('hoovertowernight.jpg'),[sz,sz])) - datamean;
styleimg = rgb2bgr(styleimg);
contextimg = rgb2bgr(contextimg);

%% get net
net = Solver.solver_.net;
layernames = net.layer_names();
len = length(layernames);

%% set style and context 
stylelayernames={'conv2_1','conv3_1'};
styleblobnames = stylelayernames;
contextlayernames={'conv3_1'};
contextblobnames = contextlayernames;



%% get context feature
contextfeature={};
net.blobs('data').set_data(contextimg);
for i = 1:len
    net.forward_fromto(i-1,i-1);
end
for i = 1:length(contextblobnames)
    contextfeat = net.blobs(contextblobnames{i}).get_data();
    contextfeature{i}=contextfeat;
end


%% get style feature
stylefeature={};
net.blobs('data').set_data(styleimg);
for i = 1:len
    net.forward_fromto(i-1,i-1);
end
for i = 1:length(styleblobnames)
    stylefeat = net.blobs(styleblobnames{i}).get_data();
    stylefeature{i}=stylefeat;
end



%% make random input
randinput=rand(sz,sz,3)*255 - 128;




%% start do style or context transform
for iter=1:10000
    %% set input data, then fp
    net.blobs('data').set_data(randinput);
    net.forward_fromto(0,layerid-1);    
    out = net.blobs(blobname).get_data();
    
    %% get loss and delta
    [delta loss]= L2Loss(out,contextfeat);
    fprintf('loss=%d\n',loss);
    
    %% set delta, then bp
    net.blobs(layername).set_diff(delta);
    Solver.solver_.net.backward_fromto(layerid-1,0);
    datadelta = net.blobs('data').get_diff();
   
    %% update input
    randinput = randinput - 1e-4* datadelta;
   

   %% show
   if mod(iter,50)==0
       show = randinput;
       show = show - min(show(:));
       show = show ./ max(show(:));
       imshow(show,[])
       max(datadelta(:))
       max(randinput(:))
       disp(iter)
   end

end


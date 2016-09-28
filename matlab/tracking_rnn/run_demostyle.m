
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
contextlayernames={'conv2_1','conv3_1','conv4_1'};
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
    
    
    %% style
    if length(stylelayernames)>0
        
        style_input_delta=zeros(size(randinput));
        style_loss = 0;
        net.blobs('data').set_data(randinput);
        for i=1:length(stylelayernames)
            %% fp
            layer_id = layer_name2id(layernames,stylelayernames{i});
            net.forward_fromto(0,layer_id);
            act = net.blobs(styleblobnames{i}).get_data();
            %% loss
            [delta loss]= styleloss(act,stylefeature{i});

            %% set delta, then bp
            net.blobs(styleblobnames{i}).set_diff(delta);
            Solver.solver_.net.backward_fromto(layer_id-1,0);
            input_delta = net.blobs('data').get_diff();
            style_input_delta = style_input_delta + input_delta;
            style_loss = style_loss + loss;
        end
        style_input_delta =style_input_delta ./ length(stylelayernames);
        style_loss = style_loss / length(stylelayernames);
        
    end
    
    %% context
    if length(contextlayernames)>0
        
        context_input_delta=zeros(size(randinput));
        context_loss = 0;
        net.blobs('data').set_data(randinput);
        for i=1:length(contextlayernames)
            %% fp
            layer_id = layer_name2id(layernames,contextlayernames{i});
            net.forward_fromto(0,layer_id);
            act = net.blobs(contextblobnames{i}).get_data();
            %% loss
            [delta loss]= L2Loss(act,contextfeature{i});

            %% set delta, then bp
            net.blobs(contextblobnames{i}).set_diff(delta);
            Solver.solver_.net.backward_fromto(layer_id-1,0);
            input_delta = net.blobs('data').get_diff();
            context_input_delta = context_input_delta + input_delta;
            context_loss = context_loss + loss;
        end
        context_input_delta =context_input_delta ./ length(contextlayernames);
        context_loss = context_loss / length(contextlayernames);
        
    end
    
    fprintf('style loss = %d, context loss = %d\n',style_loss,context_loss);
 
   
    %% update input
    randinput = randinput - 1e-4* (0.5* context_input_delta + 0.0 * style_input_delta);
   

   %% show
   if mod(iter,20)==0
       show = randinput;
       show = show - min(show(:));
       show = show ./ max(show(:));
       imshow(show,[])
       disp(iter)
   end

end


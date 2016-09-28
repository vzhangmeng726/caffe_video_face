%%
clear all 
close all


warning off
mex_path = '/home/liangji/workspace/caffe/bitbucket/caffe-liangji/matlab/+caffe';
addpath(mex_path);
addpath('/home/liangji/workspace/caffe/bitbucket/caffe-liangji/matlab');
addpath('/home/liangji/workspace/caffe/bitbucket/caffe-liangji/matlab/..');
addpath('/home/liangji/workspace/caffe/bitbucket/caffe-liangji/matlab/util');

trainname='./vot2014';
train_id=1;



%% Init
solver_file = fullfile(trainname, sprintf('%.2d',train_id), sprintf('solver_%.2d.prototxt',train_id));
STR = SolverParser(solver_file);
STR.Solver_.net.copy_from('./initmodel/VGG_CNN_M.caffemodel');
caffe.set_mode_gpu();


STR = Data_config(STR);

%% get net
net = STR.Solver_.net;
layernames = net.layer_names();
len = length(layernames);




outputblobname='output';
output_layer_id = layer_name2id(layernames,'output');

zerosinput = single(zeros(52,52,256,1));

for iter=1:STR.max_iter
    
    %video_id = mod(int32(rand * 1000),length(STR.track_list))+1;
    
    [batch, bbox, STR] = GenerateSample_rnn(STR, 'pretrain');
    inputdata = batch{1};
  
    net.blobs('data').set_data(inputdata);
    net.blobs('MGU_mgu1_h0').set_data(zerosinput);
    
    
    
    net.forward_fromto(0,output_layer_id-1);
    
    active_ = net.blobs(outputblobname).get_data();
    
    if mod(iter , 10)==0
        a=active_(:,:,STR.video_id,20);
        imshow(a)
        im=inputdata(:,:,:,20);
        im=permute(im,[2,1,3]);
        a = repmat(a,[1,1,3]);
        bb=round(bbox(20,:));
        im = uint8(im-min(im(:)));
        im(bb(4),bb(3):bb(5),1)=255;
        im(bb(2),bb(3):bb(5),1)=255;
        im(bb(4):bb(2),bb(3),1)=255;
        im(bb(4):bb(2),bb(5),1)=255;
        
        a=imresize(a,[size(im,1),size(im,2)],'nearest');
        
        
        
        a(bb(4),bb(3):bb(5),1)=255;
        a(bb(2),bb(3):bb(5),1)=255;
        a(bb(4):bb(2),bb(3),1)=255;
        a(bb(4):bb(2),bb(5),1)=255;
        
        subplot(1,2,1); imshow(a);
         subplot(1,2,2);imshow(im);
    end
    
    
    [delta_,loss_,iou_] = SigmSampledLoss_Single(active_, bbox, STR, 'pretrain');
    fprintf(' iter: %d, loss:%d, iou:%d\n',iter,loss_,iou_);
    
    net.blobs(outputblobname).set_diff(delta_);
    STR.Solver_.net.backward_fromto(output_layer_id-1,0);
            
    STR.Solver_.update();
    %print_datadiff(STR.Solver_.net);
    
    STR.Solver_.set_iter(iter);
    
    if mod(iter,STR.snapshot)==0
        savename = sprintf('%s_%d.caffemodel',STR.snapshot_prefix,iter);
        STR.Solver_.net.save(savename);
    end
  

end


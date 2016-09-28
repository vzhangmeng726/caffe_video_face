function [batch label]=get_batch(datalist,inputsize,outputsize,inputchannel,outputchannel,batchsize)
% inputsize=64;
% outputsize=32;
% inputchannel=1;
% outputchannel=2;
% batchsize=10;
shuffle=1;
batch = single(zeros(inputsize,inputsize,inputchannel,batchsize));
label = single(zeros(outputsize,outputsize,outputchannel,batchsize));
train_num = length(datalist);
count=1;
idpool=1:train_num;
if shuffle
    idpool = randperm(train_num);
end
while count<batchsize
    idx = idpool(count);
    imname = datalist(idx).im;
    labelname = datalist(idx).lb;
    img=imread(imname);
    lab=imread(labelname);
    
    img = imresize(img,[inputsize,inputsize],'bilinear');
    lab = imresize(lab,[inputsize,inputsize],'nearest');
    
    [r,c, ~] = size(img);
   % jittering
   rate =  (rand - 0.5)/5;
   shift_x = floor(max(r,c) * rate);
   rate =  (rand - 0.5)/5;
   shift_y = floor(max(r,c) * rate);
   scale_x = 1+(rand-0.5)/5;
   scale_y =  scale_x;
   angle = (rand - 0.5)*(30/180)*pi;
   A = [scale_x * cos(angle), scale_y * sin(angle), shift_x;...
    -scale_x * sin(angle), scale_y * cos(angle), shift_y]';
   T = maketform('affine', A);
   simg = single(imtransform(img, T, 'XYScale',1));
   slb = single(imtransform(lab, T,'nearest', 'XYScale',1,  'FillValues', 0));
   
   simg = imresize(simg, [inputsize,inputsize])/255-0.5;
   slb = imresize(slb, [outputsize,outputsize], 'nearest');
   
   %% get expand label
    tlb = zeros(size(slb,1),size(slb,2),outputchannel);
    for c=1:outputchannel
        temp = zeros(size(tlb,1),size(tlb,2));
        idx = slb==c-1;
        temp(idx)=1;
        tlb(:,:,c)=temp;
    end
    
   batch(:,:,:,count) = simg;
   label(:,:,:,count) = tlb;
   count = count +1;
end
batch = BatchProcess(batch);
end
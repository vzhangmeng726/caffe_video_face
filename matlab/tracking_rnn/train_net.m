


warning off
mex_path = '../+caffe';
addpath(mex_path);
addpath('..');
addpath('../..');
addpath('../util');

solver_file = './model_eye/solver.prototxt';
datafolder='/data/eye_parsing/0601/open-eye/';
listfile='/data/eye_parsing/0601/list.txt';
inputsize=64;
outputsize=64;
inputchannel=1;
outputchannel=2;
batchsize=10;

Solver = SolverParser(solver_file);
%Solver.solver_.net.copy_from('./model_eye/eye_parsing_95000.caffemodel');

traindata = get_parsing_data(datafolder,listfile);
caffe.set_mode_gpu();


for iter=1:Solver.max_iter
   
   [batch label]=get_batch(traindata,inputsize,outputsize,inputchannel,outputchannel,batchsize);
   label = permute(label,[2,1,3,4]);
   active = Solver.solver_.net.forward(batch);
   
    active_ = active{1};
    delta{1} = zeros(size(active{1}));

    [delta_, loss_] = L2Loss(active_, label);
    Solver.loss(iter) = loss_;
    delta{1} = delta_;
   
    Solver.solver_.net.backward(delta);
    
    Solver.solver_.update();
    
    Solver.solver_.set_iter(iter);
    
    if mod(iter,10)==0
        fprintf('loss for iter %.4d : %d\n', iter, Solver.loss(iter));
    end
    if mod(iter,20)==0
        
        a=active{1};
        b=label;
        d=batch{1};
        id=1;
        ac = a(:,:,:,id);
        [~,m]=max(ac,[],3);
         subplot(1,4,1);imshow(a(:,:,1,id),[]);
         subplot(1,4,2);imshow(m./max(m(:)),[]);
         subplot(1,4,3);imshow(b(:,:,1,id),[]);
         subplot(1,4,4);imshow(d(:,:,1,id),[]);
         pause(0.01);
     
    end
%     if mod(iter,1000)==0
%         print_param(Solver.solver_.net);
%         print_datadiff(Solver.solver_.net);
%     end
    
    if mod(iter,Solver.snapshot)==0
       savename = sprintf('%s_%d.caffemodel',Solver.snapshot_prefix,iter);
       Solver.solver_.net.save(savename);
    end
end



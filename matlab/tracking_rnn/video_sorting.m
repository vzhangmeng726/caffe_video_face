function [score_dist, distr] = video_sorting(STR)

if STR.Solver.iter < 1000
    fprintf('please wait for more iters.\n');
    score_dist = 0; distr = [];
    return
else
    scores = zeros(1,length(STR.train_list));
    bins = ones(1,length(STR.train_list));
    for iter = 1:length(STR.Solver.loss)
        video_id =  STR.Solver.vid(iter);
        bins(video_id) = bins(video_id) + 1;
        scores(video_id) = scores(video_id) + STR.Solver.iou(iter);
    end
    score_dist = scores ./ bins; % mean iou for each video
    score_dist(score_dist==0) = 1e-4;
%     score_dist_norm = (score_dist - min(score_dist))/...
%         (max(score_dist) - min(score_dist));
    max_ = max(score_dist);
    distr = [];
    for m = 1: length(STR.train_list)
        distr = [distr, repmat(m,[1,ceil(max_/score_dist(m))])];
    end
end

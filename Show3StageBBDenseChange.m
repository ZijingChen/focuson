addpath(genpath(pwd));
params = defaultParams([pwd '/']);
params.scale_ratio = 0.5;
params.inter_threshold = 0.3;
params.seedRatio = 0.2;
params.castStd = 0.2;% 0.2 for Gaussian-cast sample method
params.off_bound = 0.3; % for uni-cast sample method
params.nCastSamples = 5;
params.noutBoxes = 5;% 5; the number of output wins, should be smaller than (params.distribution_windows*params.seedRatio)
params.nRandWIns = 30; % should be 1000*****!!!
params.castMethod = 'Uni';% Uni

params.reshape.step_ratio = 0.1;
params.reshape.nRound = 1;

% params.reshape.alpha_b_epd = 0.04;% 0.01
% params.reshape.alpha_b_shrk = 0.006; %0.05 best 8 img paras
params.reshape.alpha_b_epd = 0.01;% 0.01
params.reshape.alpha_b_shrk = 0.05; %0.05 best 8 img paras
out_folder_name = 'single_obj_list';
output_root = ['./result_2018/',out_folder_name,'/'];%'./result_2018/2obj-2nd-trial/';
if ~exist(output_root)
    mkdir(output_root);
end
voc07_root = '/home/zichen/Data/MatlabPrograms/ExptOnVOC2007/VOC2007Imgs/';

load('/data/zichen/MatlabPrograms/ExptOnVOC2007/VOC2007index/cell_test.mat');
%Index_TestImgs = cell_test{2,1};% the index of two objects
single_obj_list = [881,3512,4444,5602,5621,7948] ;% really single? check the ground truth!
fail_list = [1296,1698,2464,3235,4214, 4444, 6951]; % two objs are too close, cannot seperate them
list1 = [1,69,286,388,437,629,630,718,836,881,925,928,...
    1031,1285,1589,1926,2105,2160,2216,2438,2652,...
    3295,3402, 3562, 3612,3922,3995,4112,5289,...
    6491,6535,6557,6717,7142,7371,7515,7599,...
    8118,8827,8875,9300,9346];
list2 = [286,378,437,574,925,928,1031,1285,1502,1926,...
    2160,2485,2623,2652,2806,4124,4375,4669,4757,4888,...
    5133,5724,5857,6491,7515,7599,8118,9346];
good_list = [3995,9300,1502,5133,5724,5857,7515];

Index_TestImgs = [7948];%single_obj_list;%intersect(list1,list2);
for idx_test=1:size(Index_TestImgs,2)
    %disp(num2str(Index_TestImgs(idx_test)));
    img_ind = Index_TestImgs(idx_test);
    nOutSamples = 5;
    
    img_path = [voc07_root , num2str(img_ind,'%06d'),'.jpg'];
    img = imread(img_path);
    
    % generate random boxes  on the image
    img = gray2rgb(img);
    [row, col, dim] = size(img);
    sz_img = [col, row];% in x-y coordinate
    randWins = genSamples_wholeImg( sz_img,params );
    % load weights
    
    % Both not rep, Single OBj 30 samples
    % load weights
    load('/data/zichen/MatlabPrograms/ExptOnVOC2007/WholeDataset/GenWeights/RepBorderForTrainTest/GenWeightsOnWholeTrainSet/Data/TrainingData/WL.mat');
    load('/data/zichen/MatlabPrograms/ExptOnVOC2007/WholeDataset/GenWeights/RepBorderForTrainTest/GenWeightsOnWholeTrainSet/Data/TrainingData/WU.mat');
    params.WL = WL;
    params.WU = WU;
    
    
    % generate saliency map
    ImgMapStruct = computeSalMap( img, params );
    OutBBs = BBsLURS( img, randWins,ImgMapStruct, params);
    
    % calculate objectness score
    scores_LU_PrevRS = judgeObjectness(img,params.cues,params,OutBBs,ImgMapStruct);
    scores_RandWin = judgeObjectness(img,params.cues,params,randWins,ImgMapStruct);
    LU_PrevRS_kboxes = [OutBBs,scores_LU_PrevRS];
    RandWin_kboxes = [randWins,scores_RandWin];
    
    % pick out top5 windows
    
    boxes_LU_PrevRS = PickTopRankWin(LU_PrevRS_kboxes, nOutSamples);
    boxes_RandWin = PickTopRankWin(RandWin_kboxes,  nOutSamples);
    
    
    %% get BB dense
    c_allrand = ShowBBdense( img, RandWin_kboxes(:,1:4 )   );
    c_allreshaping = ShowBBdense( img, LU_PrevRS_kboxes(:,1:4 )   );
    c_rand = ShowBBdense( img, boxes_RandWin(:,1:4 )   );%,'Distribution of random wins'
    c_reshaping= ShowBBdense( img, boxes_LU_PrevRS(: , 1:4 ) );%   'Distribution after reshaping'  );
    
    load curCMoving.mat;
    load curboxesSeedBoxes.mat;
    load curCAllMoving.mat;
    
    figure,
    subplot('Position',[0.02 0.65 0.3 0.25]), imshow(img), drawBoxes_nc(boxes_RandWin );%title('Top 5 random','fontsize',15);
    subplot('Position',[0.35 0.65 0.3 0.25]), imshow(img),drawBoxes_nc(boxes_seedBoxes );%title('Top 5 self-moving','fontsize',15);
    subplot('Position',[0.67 0.65 0.3 0.25]), imshow(img),drawBoxes_nc(boxes_LU_PrevRS );%title('Top 5 reshaping','fontsize',15);
    
    subplot('Position',[0.02 0.35 0.3 0.25]), imshow(c_allrand );%title('Top 5 random','fontsize',15);
    subplot('Position',[0.35 0.35 0.3 0.25]), imshow(c_allmoving );%title('Top 5 self-moving','fontsize',15);
    subplot('Position',[0.67 0.35 0.3 0.25]), imshow(c_allreshaping );%title('Top 5 reshaping','fontsize',15);
    
    PlotRes.boxes_RandWin = boxes_RandWin;
    PlotRes.boxes_seedBoxes = boxes_seedBoxes;
    PlotRes.boxes_LU_PrevRS = boxes_LU_PrevRS;
    
    PlotRes.c_allrand = c_allrand;
    PlotRes.c_allmoving = c_allmoving;
    PlotRes.c_allreshaping = c_allreshaping;
    
    figure_name = [output_root, num2str(Index_TestImgs(idx_test),'%06d'),'.png'];
    data_name = [output_root, num2str(Index_TestImgs(idx_test),'%06d'),'.mat'];
    saveas(gcf,figure_name)
    save(data_name, 'PlotRes');
    close all;
end





close all;
clear all;
clc;
addpath(genpath(pwd));
params = defaultParams([pwd '/']);
params.scale_ratio = 0.5;
params.inter_threshold = 0.3;
params.seedRatio = 0.2;
params.castStd = 0.2;% 0.2 for Gaussian-cast sample method
params.off_bound = 0.3; % for uni-cast sample method
params.nCastSamples = 5;
params.noutBoxes = 5;% 5; the number of output wins, should be smaller than (params.distribution_windows*params.seedRatio)
params.nRandWIns = 1000; % should be 1000*****!!!
params.castMethod = 'Uni';% Uni

params.reshape.step_ratio = 0.1;
params.reshape.nRound = 1;
params.reshape.alpha_b = 0.05;

nOutSamples = 5;

% % img tiger
 img = imread('../005279.png');% groundtruth should be [271, 256,446,348]
%  % use input boxes
%  inbox = [300,200,400,350;...
%  100, 100,446,348;...
%  100, 100,500,380;...
%  271, 256,510,350;...
%  250,290,510,320];% [zhong zi;l-u larger;four corner larger; r-d larger; bad example];


% else use random boxes
img = gray2rgb(img);
[row, col, dim] = size(img);
sz_img = [col, row];% in x-y coordinate
randWins = genSamples_wholeImg( sz_img,params );
%end if

% DEBUG ONLY
% figure;
% imshow(img);title(['org loc : ', mat2str(inbox)],'fontsize',15);
% drawBoxes_nc(inbox);
% END FOR DEBUG ONLY

% load weights

 % Both not rep, Single OBj 30 samples
load('/data/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/BothNotRep/SIngleObj30Sample/WL.mat');
load('/data/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/BothNotRep/SIngleObj30Sample/WU.mat');

% load('/home/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/withSimpleBG/WL.mat');
% load('/home/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/withSimpleBG/WU.mat');
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
boxes_LU_PrevRS = nms_pascal(LU_PrevRS_kboxes, 0.5, nOutSamples);

boxes_RandWin = nms_pascal(RandWin_kboxes, 0.5, nOutSamples);

% DEBUG ONLY
figure;
imshow(img);title('LU_PrevRS loc : ','fontsize',15);
drawBoxes_nc(  boxes_LU_PrevRS(: , 1:4 )   );

figure;
imshow(img);title('RandWin loc : ','fontsize',15);
drawBoxes_nc(  boxes_RandWin(:,1:4 )   );




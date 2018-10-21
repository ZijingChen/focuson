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

nOutSamples = 20;

% % img tiger
 img = imread('005279.png');% groundtruth should be [271, 256,446,348]
 GT =  [271, 256,446,348];
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
load('/home/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/BothNotRep/SIngleObj30Sample/WL.mat');
load('/home/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/BothNotRep/SIngleObj30Sample/WU.mat');

% load('/home/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/withSimpleBG/WL.mat');
% load('/home/zichen/MatlabPrograms/ExptOnVOC2007/AllWeights/withSimpleBG/WU.mat');
params.WL = WL;
params.WU = WU;


% generate saliency map
ImgMapStruct = computeSalMap( img, params );

% get LU-shift box
IntgSalMat = ImgMapStruct(1,5).salmapIntegralImageMat;
IntgThrMat = ImgMapStruct(1,5).thrmapIntegralImageMat;

[dnx, dny] = judgeOffset(img,randWins,params);
seedBoxes = genOffsetBoxes(randWins,dnx, dny,sz_img);

% calculate objectness scores
scores_LU = judgeObjectness(img,params.cues,params,seedBoxes,ImgMapStruct);
scores_RandWin = judgeObjectness(img,params.cues,params,randWins,ImgMapStruct);
RandWin_kboxes = [randWins,scores_RandWin];

% pick out top5 windows for random wins
boxes_RandWin = nms_pascal(RandWin_kboxes, 0.5, 5);


% pick out top20 windows FOR LU-SHIFT wins
seedBoxes4Rank = [seedBoxes,scores_LU];
boxes_LUtop20 = nms_pascal(seedBoxes4Rank, 0.5, nOutSamples);





% reshaping top20 windows
[nBB,~] = size(boxes_LUtop20);
%      % DEBUG ONLY
%     figure;
%     imshow(img);title(['lu loc : ', mat2str(seedBoxes)],'fontsize',15);
%     drawBoxes_nc(seedBoxes);
%     % END FOR DEBUG ONLY
     
     % reshape params preparation
     boxcolor = [1 1 0];
     OutBBs = zeros(nBB,4);
    for idx_BB = 1: nBB
        OutBBs(idx_BB,:) = ReshapeBB( boxes_LUtop20(idx_BB,1:4), img, IntgSalMat, IntgThrMat,params,boxcolor ); 
    end
    
    % pick out top5 windows after reshaping
    scores_LU_PostRS = judgeObjectness(img,params.cues,params,OutBBs,ImgMapStruct);
    OutBBs4Rank = [OutBBs, scores_LU_PostRS];
    boxes_LU_PostRS = nms_pascal(OutBBs4Rank, 0.5, 5);

% test performance
% % test1, with out normalization cle
% [ CLE_LUtop5, Overlap_LUtop5 ] = GetCleOl( boxes_LUtop20(1:5,1:4), GT );
% [ CLE_LU_PostRS, Overlap_LU_PostRS ] = GetCleOl( boxes_LU_PostRS(:,1:4), GT );
% [ CLE_RandWin, Overlap_RandWin ] = GetCleOl( boxes_RandWin(:,1:4), GT );
% 
% display('Top 5 random windows');
% mean_CLE_RandWin = mean(CLE_RandWin,1)
% mean_Overlap_RandWin = mean(Overlap_RandWin,1)
% 
% display('Top 5 LU shift windows');
% mean_CLE_LUtop5 = mean(CLE_LUtop5,1)
% mean_Overlap_LUtop5 = mean(Overlap_LUtop5,1)
% 
% display('Top 5 LU and reshape windows');
% mean_CLE_LU_PostRS = mean(CLE_LU_PostRS,1)
% mean_Overlap_LU_PostRS = mean(Overlap_LU_PostRS,1)
% 
% % end test1

% test 2
 GTStruct(1,1).boxes = GT;
ResStruct_RANDWIN(1,1).boxes = boxes_RandWin(:,1:4);

ResStruct_LU(1,1).boxes = boxes_LUtop20(1:5,1:4);

ResStruct_LURS(1,1).boxes = boxes_LU_PostRS(:,1:4);

[ PerformStruct_RANDWIN, Ave_RANDWIN ] = JudgeStructRes( ResStruct_RANDWIN, GTStruct )
[ PerformStruct_LU, Ave_LU ] = JudgeStructRes( ResStruct_LU, GTStruct )
[ PerformStruct_LURS, Ave_LURS ] = JudgeStructRes( ResStruct_LURS, GTStruct )

% BEGIN debug only
figure,
subplot(1,3,1);
imshow(img);title('Best 5 RandWin loc : ','fontsize',15);
drawBoxes_nc(  boxes_RandWin(:,1:4 )   );

% END DEBUG ONLY

% debug only
subplot(1,3,2);
imshow(img);title('Best 5 LU shift loc : ','fontsize',15);
drawBoxes_nc(  boxes_LUtop20 (1:5 , 1:4 )   );


%end debug only

% DEBUG ONLY
subplot(1,3,3);
imshow(img);title('Best 5 LU and post reshape loc : ','fontsize',15);
drawBoxes_nc(  boxes_LU_PostRS (: , 1:4 )   );
% END DEBUG ONLY

%% show BB dense
Flag1 = ShowBBdense( img, boxes_RandWin(:,1:4 )  );
Flag2 = ShowBBdense( img, boxes_LUtop20 (1:5 , 1:4 )    );
Flag3 = ShowBBdense( img, boxes_LU_PostRS (: , 1:4 )   );

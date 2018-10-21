addpath(genpath(pwd));
warning off
clear;

% loading general parameters
display('Loading the default parameters ...');
param_root = '/data/zichen/MatlabPrograms/ExptOnVOC2007/Validation/ValdLURS02_NewRsMethod'
params = defaultParams(param_root);
params.scale_ratio = 0.5;
params.inter_threshold = 0.3;
params.seedRatio = 0.2;
params.castStd = 0.2;% 0.2 for Gaussian-cast sample method
params.off_bound = 0.3; % for uni-cast sample method
params.nCastSamples = 5;
params.noutBoxes = 5;% 5; the number of output wins, should be smaller than (params.distribution_windows*params.seedRatio)
params.nRandWIns = 30; % should be 1000*****!!!
params.castMethod = 'Uni';% Uni
params.img_folder = '/data/zichen/MatlabPrograms/ExptOnVOC2007/VOC2007Imgs/';

params.reshape.step_ratio = 0.1;
params.reshape.nRound = 1;
params.reshape.alpha_b_epd = 0.01;
params.reshape.alpha_b_shrk = 0.05;
params.Lu2RsSamples = 20;

%%% loading data
load('VOC2007_LocsNname.mat');% the name is  'VOC2007_struct'
structGT = VOC2007_struct;

% load('/home/zichen/MatlabPrograms/ExptOnVOC2007/VOC2007index/test.mat');
load('/data/zichen/MatlabPrograms/ExptOnVOC2007/VOC2007index/cell_test.mat');
Index_TestImgs = cell_test{1,1};
% Index_TestImgs = Index_TestImgs(1,1:5); % debug only

% load weights
load('/data/zichen/MatlabPrograms/ExptOnVOC2007/WholeDataset/GenWeights/RepBorderForTrainTest/GenWeightsOnWholeTrainSet/Data/TrainingData/WL.mat');
load('/data/zichen/MatlabPrograms/ExptOnVOC2007/WholeDataset/GenWeights/RepBorderForTrainTest/GenWeightsOnWholeTrainSet/Data/TrainingData/WU.mat');

params.WL = WL;
params.WU = WU;


GT = structGT(Index_TestImgs);


%% generate 1000 random boxes and offset boxes and draw hist figures
[ OrgStruct, LuStruct, RsStruct ] = GenOrgLuRsBoxes( GT,params );
save OrgStruct_1obj.mat OrgStruct;
save LuStruct_1obj.mat LuStruct;
save LuRsStruct_30samples_1obj.mat RsStruct;
save GT.mat GT;
RunPerformanceJudge;
%******END generate 1000 random boxes and ..hist figures************

%% generate objectness predictions and calculate cle with ol
% ObjRes_Struct = GenObjectnessBoxes( GT,params );
% [ Perform_LURS, Ave_LURS] = JudgeStructRes( ObjRes_Struct, GT  );

% show performance with histograms

% figure,
% subplot(2,1,1),hist(ResStruct.CLE_ork,390),title('CLE of K origin boxes');
% axis([0,1,0,1000]);
% subplot(2,1,2),hist(ResStruct.CLE_ofk,390 ),title('CLE of K offset boxes');
% axis([0,1,0,1000]);
% figure,
% subplot(2,1,1),hist(ResStruct.CLE_or5,350 ),title('CLE of 5 origin boxes');
% axis([0,1,0,200]);
% subplot(2,1,2),hist(ResStruct.CLE_of5,350),title('CLE of 5 offset boxes');
% axis([0,1,0,200]);
% 
% figure,
% subplot(2,1,1),hist(ResStruct.ORIGIN_5boxes,50),title('score of 5 origin boxes');
% axis([0,1,0,150]);
% subplot(2,1,2),hist(ResStruct.OFFSET_5boxes,50),title('score of 5 offset boxes');
% axis([0,1,0,150]);
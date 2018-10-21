function [ OrgStruct, LuStruct, RsStruct ] = GenOrgLuRsBoxes( GT, params )
%UNTITLED2 Summary of this function goes here
%  GENERATE RESHAPE WINDOW FOR each LU WINDOW
nimg = size(GT,2);
Lu2RsSamples = params.Lu2RsSamples;%  = 20;
noutBoxes = params.noutBoxes ; % = 5

for idx_img =1:nimg
    img = imread( [params.img_folder,GT(1,idx_img).img ] );
    display(['Deal with image: ', GT(1,idx_img).img]);

    % generate all possible random boxes here
    img = gray2rgb(img);
    [row, col, dim] = size(img);
    sz_img = [col, row];% in x-y coordinate
    randWins = genSamples_wholeImg( sz_img,params );
    
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
    RandWin_kboxes = [randWins,scores_RandWin]; %~~~
    
    % pick out top5 windows for random wins
%     boxes_RandWin = nms_pascal(RandWin_kboxes, 0.5, noutBoxes); % choose NMS pick
    boxes_RandWin = PickTopRankWin(RandWin_kboxes, noutBoxes); %  else choose des pick
    

    % pick out top5 windows FOR LU-SHIFT wins
    seedBoxes4Rank = [seedBoxes,scores_LU];
%     boxes_LUtop5 = nms_pascal(seedBoxes4Rank, 0.5, noutBoxes); % NMS pick out 5~~
    boxes_LUtop5 = PickTopRankWin(seedBoxes4Rank, noutBoxes); % or des pick out 5~~
    
    [nBB,~] = size(seedBoxes4Rank);
    % reshape params preparation
    boxcolor = [1 1 0];
    OutBBs = zeros(nBB,4);
    for idx_BB = 1: nBB
        OutBBs(idx_BB,:) = ReshapeBB( seedBoxes4Rank(idx_BB,1:4), img, IntgSalMat, IntgThrMat,params,boxcolor );
    end
    
    % pick out top5 windows after reshaping
    scores_LU_PostRS = judgeObjectness(img,params.cues,params,OutBBs,ImgMapStruct);
    OutBBs4Rank = [OutBBs, scores_LU_PostRS];
%     boxes_LU_PostRS = nms_pascal(OutBBs4Rank, 0.5, noutBoxes); % NMS pick out 5~~~~~
    boxes_LU_PostRS = PickTopRankWin(OutBBs4Rank,noutBoxes); % or des pick out 5~~~~~
    
    
    % preserve result in 3 structures
    OrgStruct(1, idx_img).totalBoxes_loc = randWins; % locs for 1000 windows
    OrgStruct(1, idx_img).totalBoxes_score = scores_RandWin;% score for 1000 windows
    OrgStruct(1, idx_img).top5Boxes_loc = boxes_RandWin(:,1:4);  % locs for top 5 windows selected from random windows
    OrgStruct(1, idx_img).top5Boxes_score = boxes_RandWin(:,5);% score for TOP5 windows
    
    LuStruct(1, idx_img).totalBoxes_loc  = seedBoxes;
    LuStruct(1, idx_img).totalBoxes_score = scores_LU;
    LuStruct(1, idx_img).top5Boxes_loc = boxes_LUtop5(:,1:4);  % locs for top 5 windows selected from LU windows
    LuStruct(1, idx_img).top5Boxes_score = boxes_LUtop5(:,5);% score for TOP5 windows
    
    RsStruct(1, idx_img).totalBoxes_loc  = OutBBs;
    RsStruct(1, idx_img).totalBoxes_score = scores_LU_PostRS;
    RsStruct(1, idx_img).top5Boxes_loc = boxes_LU_PostRS(:,1:4);  % locs for top 5 windows selected from LU windows
    RsStruct(1, idx_img).top5Boxes_score = boxes_LU_PostRS(:,5);% score for TOP5 windows
    
end % end for each image

end


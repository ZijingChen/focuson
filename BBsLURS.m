function OutBBs = BBsLURS( img, inBBs,ImgMapStruct, params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    img = gray2rgb(img);
    [row, col, dim] = size(img);
    sz_img = [col, row];% in x-y coordinate
   
      IntgSalMat = ImgMapStruct(1,5).salmapIntegralImageMat;
    IntgThrMat = ImgMapStruct(1,5).thrmapIntegralImageMat;
     
     [dnx, dny] = judgeOffset(img,inBBs,params);
     
     seedBoxes = genOffsetBoxes(inBBs,dnx, dny,sz_img);
     [nBB,~] = size(seedBoxes);
     % DEBUG ONLY
    
    scores_seedBoxes = judgeObjectness(img,params.cues,params,seedBoxes,ImgMapStruct);
    seedBoxes_forRank = [ seedBoxes, scores_seedBoxes ];
    %%%boxes_seedBoxes = nms_pascal(seedBoxes_forRank,0.9, 5);
    boxes_seedBoxes = PickTopRankWin(seedBoxes_forRank, 5);
%     subfigure()imshow(img);title(['BBs after self-moving '],'fontsize',15);
%     drawBoxes_nc(boxes_seedBoxes);
    c_moving = ShowBBdense( img, boxes_seedBoxes(:,1:4 ) );% ,'Distribution after self-moving'   
    c_allmoving = ShowBBdense( img, seedBoxes_forRank(:,1:4 ) );
    save curCMoving.mat c_moving;
    save curCAllMoving.mat c_allmoving;
    save curboxesSeedBoxes.mat boxes_seedBoxes
    % END FOR DEBUG ONLY
     tic
     % reshape params preparation
     boxcolor = [1 1 0];
     OutBBs = zeros(nBB,4);
    for idx_BB = 1: nBB
        OutBBs(idx_BB,:) = ReshapeBB( seedBoxes(idx_BB,:), img, IntgSalMat, IntgThrMat,params,boxcolor );
        
    end
toc

end


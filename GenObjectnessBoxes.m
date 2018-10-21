function ObjRes_Struct = GenObjectnessBoxes( GT,params )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
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

        % pick out top5 windows FOR LU-SHIFT wins
        seedBoxes4Rank = [seedBoxes,scores_LU];
        boxes_LUtop20 = nms_pascal(seedBoxes4Rank, 0.5, Lu2RsSamples); % pick out 20~~

        [nBB,~] = size( boxes_LUtop20);
        % reshape params preparation
        boxcolor = [1 1 0];
        OutBBs = zeros(nBB,4);
        for idx_BB = 1: nBB
            OutBBs(idx_BB,:) = ReshapeBB( seedBoxes4Rank(idx_BB,1:4), img, IntgSalMat, IntgThrMat,params,boxcolor );
        end

        % pick out top5 windows after reshaping
        scores_LU_PostRS = judgeObjectness(img,params.cues,params,OutBBs,ImgMapStruct);
        OutBBs4Rank = [OutBBs, scores_LU_PostRS];
        boxes_LU_PostRS = nms_pascal(OutBBs4Rank, 0.5, noutBoxes); %~~~

        % preserve result in  structures

        ObjRes_Struct (1, idx_img).boxes = boxes_LU_PostRS(:,1:4);  % locs for top 5 windows selected from random windows
        ObjRes_Struct(1, idx_img).scores =boxes_LU_PostRS(:,5);% score for TOP5 windows


    end % end for each image

end


function c= ShowBBdense( img, inBBs )
%UNTITLED Summary of this function goes here
%   figure,imshow(c);title(title_name);
    [ row, col,~ ] = size(img);
    nbox = size(inBBs,1);
    
    % sum box location
    SumDenseMat = zeros(row, col);
    
    for idx_box = 1:nbox
        currBBloc = inBBs(idx_box , :);
        xmin =  currBBloc (1,1);
        ymin =  currBBloc (1,2);
        xmax =  currBBloc (1,3);
        ymax =  currBBloc (1,4);
%         [ BBrow, BBcol ] = size(currBBloc);
        
        for idx_col = xmin:xmax
            SumDenseMat(ymin:  ymax,  idx_col) = SumDenseMat(ymin:  ymax,  idx_col)+1;
        end
        
    end
    
    % show dense matrix
    b=SumDenseMat;
    b = imresize(b,[row, col]);
    b= b-min(b(:));
    b= b/max(b(:));
    c=zeros([size(b),3]);
%     c(:,:,1) = b;%R
%     c(:,:,2) = 0;%G
%     c(:,:,3) = 0;%B %% the original is c(:,:,3) = 1-b;%B
    c(:,:,1) = b;%R
    c(:,:,2) = 0;%G
    c(:,:,3) = 0;%B %% the original is c(:,:,3) = 1-b;%B
    
    
%     MatMin = min( min(SumDenseMat) );
%     MatMax =max( max(SumDenseMat) );
%     SumDenseMat =( SumDenseMat-MatMin)./MatMax;
%     SumDenseMat =  SumDenseMat.*255;
%     % prepare for interpolation
%     ipl_row = max( row, 200) ;
%     ipl_col = max(  col ,  ceil(200*(col/row) ));
%     a = imresize(SumDenseMat,[ipl_row ,ipl_col]);
%     figure, image(a);figure(gcf);
    


end


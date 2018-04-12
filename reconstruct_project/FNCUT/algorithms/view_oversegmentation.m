function view_oversegmentation(label_img,seg_img,out_path,only_name)

[X Y Z] = size(label_img{1}); L = size(label_img,2); seg_path = [out_path 'segs/']; N = X*Y;
mkdir(seg_path);

%% make the resulted image with red boundaries
for i=1:L,
    [imgMasks,segOutline,imgMarkup]=segoutput(seg_img{i},double(label_img{i}));
    imwrite(imgMarkup,[seg_path only_name '_' int2str(i) '.bmp']); 
    clear imgMasks segOutline imgMarkup;
end;
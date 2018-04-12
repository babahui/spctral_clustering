function [gt_imgs gt_cnt] = view_gt_segmentation(img,name,out_path,only_name,save)

gt_imgs = readSegs('color',name);
gt_path = [out_path 'gt/']; mkdir(gt_path);

gt_cnt = [];
for i=1:size(gt_imgs,2), 
    if save == 1,
        [imgMasks,segOutline,imgMarkup]=segoutput(img,gt_imgs{i});
        imwrite(imgMarkup,[gt_path only_name '_' int2str(i) '.bmp']); 
    end;
    gt_cnt(i) = max(gt_imgs{i}(:));
end;

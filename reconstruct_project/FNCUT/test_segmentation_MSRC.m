
%% MSRC database
clear all; close all;

addpath 'msseg'
addpath 'others'
addpath 'algorithms';

imgRoot = 'F:\00_DATABASE\MSRC\MSRC_ObjCategImageDatabase\Images\';
nums = [30, 30, 30, 30, 30, 30, 30, 30, 30, 32, ...
        30, 34, 30, 30, 24, 30, 30, 30, 30, 21];

%%% parameters
L = 30;             % the number of the segments
scale = 0.6;        % image scale
lambda = 0.99999;   % a weighting values for smoothness terms

para.K = 3;         % the number of the over-segmention layers
para.alpha = 0.001; % the relationship between pixels and regions
para.beta  =  60;   % the variance of the color differences
para.gamma = 1.0;   % the smoothness between regions in the same layer

% meanshift image segmentation
para.hs{1} =  5; para.hr{1} = 7; para.M{1} = 30;
para.hs{2} =  7; para.hr{2} = 5; para.M{2} = 30;
para.hs{3} =  7; para.hr{3} = 7; para.M{3} = 30;

display_eig = 0;
display_seg = 0;

for category=1:20,
    out_segmentation_path = ['results/MSRC/' int2str(category) '/']; mkdir(out_segmentation_path);
    for i=1:nums(category), iids{i} = [int2str(category) '_' int2str(i) '_s']; end;
    
    for aaa=1:size(iids,2)
        only_name = iids{aaa};
        img_name = [imgRoot only_name '.bmp'];
        out_path = [out_segmentation_path only_name '/'];   mkdir(out_path);
        
        bExist_SP = exist([out_path 'data/' only_name '_' int2str(L) '_sp.mat'],'file');
        if ~bExist_SP,
            % make graph
            [W,img,label_img,seg,seg_img] = make_weight_matrix(img_name,scale,para); X = size(img,1); Y = size(img,2);
            view_oversegmentation(label_img,seg_img,out_path,only_name);
            % make eigenspace
            [B,evec,evals,DD2_i] = make_spectral_analysis(W,L,lambda,out_path,only_name);
            view_eigenspace_all(img,evec,evals,out_path,only_name,display_eig);
            clear W label_img seg seg_img;
        else
            img = im2double(imread(img_name)); img = imresize(img,scale); X = size(img,1); Y = size(img,2);
            load([out_path 'data/' only_name '_' int2str(L) '_sp.mat']);
        end;

        % spectral segmentation
        bExist_SG = exist([out_path 'data/' only_name '_' int2str(L) '_ncut.mat'],'file');
        if ~bExist_SG,
            labels = ncut_B(evec(:,1:L),DD2_i,L,X*Y,out_path,only_name);
        else
            load([out_path 'data/' only_name '_' int2str(L) '_ncut.mat']);
        end;
        out_vals = [];
        view_segmentation(img,labels,out_vals,out_path,only_name,display_seg);
        clear img B evec evals DD2_i labels;
    end;
    clear iids;
end;

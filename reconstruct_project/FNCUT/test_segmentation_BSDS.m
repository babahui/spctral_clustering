
%% BSDS300 database
clear all; close all;

addpath 'msseg'
addpath 'others'
addpath 'evals'
addpath 'algorithms';

%%% parameters
max_L = 40;         % the maximum number of the segments
scale = 1.0;        % image scale
lambda = 0.99999;   % a weighting values for smoothness terms

para.K = 3;         % the number of the over-segmention layers
para.alpha = 0.001; % the relationship between pixels and regions
para.beta  =  60;   % the variance of the color differences
para.gamma = 1.0;   % the smoothness between regions in the same layer

% meanshift image segmentation
para.hs{1} =  5; para.hr{1} = 7; para.M{1} = 100;
para.hs{2} =  7; para.hr{2} = 5; para.M{2} = 100;
para.hs{3} =  7; para.hr{3} = 7; para.M{3} = 100;

display_eig = 0; 
display_seg = 0;

% bsdsRoot = 'F:\00_DATABASE\BSDS300';
bsdsRoot = '/home/yy/hust_lab/CV/github_spectral_clustering/reconstruct_project/FNCUT/BSDS300';

fid = fopen('results/BSDS300/fncut_opt.txt','r');
[BSDS_INFO] = fscanf(fid,'%d %d \n',[2,300]);
fclose(fid);

for aaa=1:size(BSDS_INFO,2),
    L = BSDS_INFO(2,aaa);
    only_name = int2str(BSDS_INFO(1,aaa));
    img_name = [bsdsRoot '/images/test/' only_name '.jpg'];
    if ~exist(img_name,'file'), img_name = [bsdsRoot '/images/train/' only_name '.jpg']; end;
    out_path = ['results/BSDS300/' only_name '/'];   mkdir(out_path);

    bExist_OUT = exist([out_path 'data/' only_name '_' int2str(L) '_out.mat'],'file');
    if ~bExist_OUT,
        bExist_SP = exist([out_path 'data/' only_name '_' int2str(max_L) '_sp.mat'],'file');
        if ~bExist_SP,
            [W,img,label_img,seg,seg_img] = make_weight_matrix(img_name,scale,para); X = size(img,1); Y = size(img,2);
            view_oversegmentation(label_img,seg_img,out_path,only_name);
            [gt_imgs gt_cnt] = view_gt_segmentation(img,BSDS_INFO(1,aaa),out_path,only_name,1);
            % make eigenspace
            [B,evec,evals,DD2_i] = make_spectral_analysis(W,max_L,lambda,out_path,only_name);
            view_eigenspace_all(img,evec,evals,out_path,only_name,display_eig);
            clear W label_img seg seg_img;
        else
            img = im2double(imread(img_name)); img = imresize(img,scale);  X = size(img,1); Y = size(img,2);
            load([out_path 'data/' only_name '_' int2str(max_L) '_sp.mat']);
            [gt_imgs gt_cnt] = view_gt_segmentation(img,BSDS_INFO(1,aaa),out_path,only_name,0);
        end;

        bExist_SG = exist([out_path 'data/' only_name '_' int2str(L) '_ncut.mat'],'file');
        if ~bExist_SG,
            labels = ncut_B(evec(:,1:L),DD2_i,L,X*Y,out_path,only_name);
        else
            load([out_path 'data/' only_name '_' int2str(L) '_ncut.mat']);
        end;

        % update the four error measures:
        out_vals.PRI = 0; out_vals.VoI = 0; out_vals.GCE = 0; out_vals.BDE = 0;
        for i=1:size(gt_imgs,2),
            out_vals.BDE = out_vals.BDE + compare_image_boundary_error(reshape(labels,X,Y),gt_imgs{i});        
            [curRI,curGCE,curVOI] = compare_segmentations(reshape(labels,X,Y),gt_imgs{i});       
            out_vals.PRI = out_vals.PRI + curRI;
            out_vals.VoI = out_vals.VoI + curVOI;
            out_vals.GCE = out_vals.GCE + curGCE;  
        end;
        out_vals.PRI = out_vals.PRI/size(gt_imgs,2); out_vals.PRI
        out_vals.VoI = out_vals.VoI/size(gt_imgs,2); out_vals.VoI
        out_vals.GCE = out_vals.GCE/size(gt_imgs,2); out_vals.GCE
        out_vals.BDE = out_vals.BDE/size(gt_imgs,2); out_vals.BDE
        view_segmentation(img,labels,out_vals,out_path,only_name,display_seg);
        clear img B evec evals DD2_i labels;
    else
        load([out_path 'data/' only_name '_' int2str(L) '_out.mat']);
    end;
    
    PRI_all(aaa) = out_vals.PRI;
    VoI_all(aaa) = out_vals.VoI;
    GCE_all(aaa) = out_vals.GCE;
    BDE_all(aaa) = out_vals.BDE;
    clear labels out_vals;
end;

fid_out = fopen('results/BSDS300/evaluation_opt.txt','w');
for aaa=1:size(BSDS_INFO,2),
    fprintf(fid_out,'%d %2.6f %2.6f %2.6f %2.6f \n', BSDS_INFO(1,aaa), PRI_all(aaa), VoI_all(aaa), GCE_all(aaa), BDE_all(aaa));
end;
fprintf(fid_out,'%2.6f %2.6f %2.6f %2.6f \n', mean(PRI_all), mean(VoI_all), mean(GCE_all), mean(BDE_all));
fclose(fid_out);

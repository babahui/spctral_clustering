function [W,img,L,seg,seg_img] = make_weight_matrix(img_name,scale,para)

ori_img = imread(img_name); 
res_img = imresize(ori_img,scale); 
img = im2double(res_img); [X,Y,Z] = size(res_img); N = X*Y;  % image resize

lab_img = colorspace('Lab<-', res_img);
lab_valsX = reshape(lab_img, X*Y, Z);

%% make regions for each layer
if para.K == 0,
    L = []; seg = []; seg_img = [];
end;
for k=1:para.K,
    [S{k} L{k} seg{k} seg_vals{k} seg_lab_vals{k} seg_edges{k}] = msseg(double(res_img),lab_valsX,para.hs{k},para.hr{k},para.M{k});
    % make mean color image for display
    seg_img{k} = zeros(N,Z); nseg{k} = size(seg{k},2);
    for i=1:nseg{k},
        for j=1:Z, seg_img{k}(seg{k}{i},j) = seg_vals{k}(i,j)/255; end;
    end;
    seg_img{k} = reshape(seg_img{k},[X,Y,Z]);
end;

%% make edges & nodes betweem pixels : 'edgesX' 'lab_valsX'
[pointsX edgesX] = lattice(X,Y,0); clear pointsX;

%% make edges & nodes between regions: 'edgesY' 'lab_valsY'
num = N; edgesY = []; lab_valsY = [];
for k=1:para.K,
    edgesY = [edgesY; (seg_edges{k}+num)]; 
    lab_valsY = [lab_valsY ; seg_lab_vals{k}];
    num = num + nseg{k};
end;

%% total weight matrix - For each layer, use different color variance
weightsX = makeweights(edgesX,lab_valsX,para.beta);
W_X = adjacency(edgesX,weightsX,N);

% make edges between pixels and regions: 'edgesXY'
W_A = []; W_B = []; nsegs = 0;
for k=1:para.K,
    nsegs = nsegs + nseg{k};
    weightsY{k} = makeweights(seg_edges{k},seg_lab_vals{k},para.beta);
    nW_Y{k} = para.gamma*adjacency(seg_edges{k},weightsY{k},nseg{k});
    % make edges between pixels and regions: 'edgesXY'
    edgesXY{k} = [];
    for i=1:nseg{k},
        col1 = seg{k}{i}; col2 = (i+N)*ones(size(col1,1),1); 
        edgesXY{k} = [edgesXY{k} ; [col1 col2]]; clear col1 col2;
    end;
    weightsXY{k} = ones(size(edgesXY{k},1),1);
    W_T = para.alpha*adjacency(edgesXY{k},weightsXY{k},N+nseg{k});
    W_A = [W_A   W_T(1:N,N+1:N+nseg{k})];
    W_B = [W_B ; W_T(N+1:N+nseg{k},1:N)];
    clear W_T;
end;

W_Y = sparse(nsegs,nsegs); num = 0;
for k=1:para.K,
    W_Y(num+1:num+nseg{k},num+1:num+nseg{k}) = nW_Y{k}; num = num + nseg{k};
end;
W = [W_X W_A ; W_B W_Y];

% % For all layer, use same color variance
% num = N; edgesXY = [];
% for k=1:para.K,
%     for i=1:nseg{k},
%         col1 = seg{k}{i}; col2 = (i+num)*ones(size(col1,1),1); 
%         edgesXY = [edgesXY ; [col1 col2]]; clear col1 col2;
%     end;
%     num = num + nseg{k}; 
% end;
% weightsT = makeweights([edgesX; edgesY; edgesXY],[lab_valsX;lab_valsY],para.beta);
% W_T = adjacency([edgesX; edgesY; edgesXY],weightsT,num);
% 
% W_X = W_T(1:N,1:N);
% W_A = para.alpha*W_T(1:N,N+1:num);
% W_B = para.alpha*W_T(N+1:num,1:N);
% W_Y = para.gamma*W_T(N+1:num,N+1:num);
% W = [W_X W_A ; W_B W_Y];

% % For pixel or region layer, use different color variance
% weightsX = makeweights(edgesX,lab_valsX,para.beta);
% W_X = adjacency(edgesX,weightsX,N);
% 
% edgesY = edgesY - N; 
% weightsY = para.gamma*makeweights(edgesY,lab_valsY,para.beta);
% W_Y = adjacency(edgesY,weightsY,num-N);
% 
% W_A = []; W_B = [];
% for k=1:para.K,
%     edgesXY{k} = [];
%     for i=1:nseg{k},
%         col1 = seg{k}{i}; col2 = (i+N)*ones(size(col1,1),1); 
%         edgesXY{k} = [edgesXY{k} ; [col1 col2]]; clear col1 col2;
%     end;
%     weightsXY{k} = ones(size(edgesXY{k},1),1)/nseg{k};
%     W_T = para.alpha*adjacency(edgesXY{k},weightsXY{k},N+nseg{k});
%     W_A = [W_A   W_T(1:N,N+1:N+nseg{k})];
%     W_B = [W_B ; W_T(N+1:N+nseg{k},1:N)];
%     clear W_T;
% end;
% W = [W_X W_A ; W_B W_Y];

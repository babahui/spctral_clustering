function [B,evec,evals,DD2_i,ptime] = make_spectral_analysis(W,L,lambda,out_path,only_name)

N=size(W,1);
data_path = [out_path 'data/']; mkdir(data_path);

%% compute the matrix B
dd = sum(W); D = sparse(1:N,1:N,dd); clear dd;

% (1-lambda)*(D-lambda*W)^-1
dd = (1-lambda)*(D-lambda*W)\ones(N,1); dd = sqrt(dd); DD2_i = sparse(1:N,1:N,1./dd); DD2 = sparse(1:N,1:N,dd); clear dd;
B = DD2*(D-lambda*W)*DD2;
 
%% compute eigenvectors
opts.issym = 1; opts.isreal = 1; opts.disp = 0; 
opts.p = 2*L;   % added parameter
mn_set = 0;

t = cputime;
[V,ss] = eigs(B,L,mn_set,opts);
ptime = cputime-t;
s = real(diag(ss)); 
[x,y] = sort(s); 
evals = 1./abs(x); evec = V(:,y);

save([data_path only_name '_' int2str(L) '_sp.mat'], 'B','evec','evals','DD2_i');
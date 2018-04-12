function view_eigenspace_all(img,evec,evals,out_path,only_name,display)

[X Y Z] = size(img); L = size(evec,2); eig_path = [out_path 'eigs/']; N = X*Y;
mkdir(eig_path);

for i=1:L,
    disp = reshape(evec(1:N,i),X,Y); out = sc(disp,'prob_jet');
    imwrite(out, [eig_path only_name '_egv_' int2str(i) '.bmp']);    clear disp out;
end;

if display == 1,
    %% display Ncut eigenvectors
    figure;clf;set(gcf,'Position',[100,500,200*(L+2),200]);
    subplot(1,L+1,1); plot(evals);
    for i=2:L+1
        subplot(1,L+1,i);
        imagesc(reshape(evec(1:N,i-1),X,Y));axis('image');axis off;
    end
end;
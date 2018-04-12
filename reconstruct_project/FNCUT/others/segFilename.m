function [filename] = segFilename(present,uid,iid)
% function [filename] = segFileName(present,uid,iid)
%
% Return seg filename composed of components.
%
% INPUT
%	present		Presentation, one of {'gray','color'}.
%	uid		User ID.
%	iid		Image ID.
%
% OUTPUT
%	filename	Segmentation filename.
%
% David Martin <dmartin@eecs.berkeley.edu>
% January 2003

bsdsRoot = 'F:\00_DATABASE\BSDS300';
filename = fullfile(bsdsRoot,'human',present,sprintf('%d',uid),sprintf('%d.seg',iid));



===========================================================================
Learning Full Pairwise Affinities for Spectral Segmentation v1.0-2010-06-28
===========================================================================

written by 
Tae Hoon Kim  <th33@snu.ac.kr>
Kyoung Mu Lee <kyoungmu@snu.ac.kr>
Sang Uk Lee   <sanguk@ipl.snu.ac.kr>

===========================================================================
Content:
===========================================================================
This algorithm was introduced in the paper:
Tae Hoon Kim, Kyoung Mu Lee, Sang Uk Lee, "Learning Full Pairwise Affinities for Spectral Segmentation", CVPR 2010

===========================================================================
Examples:
===========================================================================
1. Experiments #1: Berkeley image database (BSDS300)

1) Download BSDS300 at http://www.cs.berkeley.edu/projects/vision/bsds.
   - "Images" & "Human segmentations"

2) Set the BSDS300 directory path to a variable "bsdsRoot" in "test_segmentation_BSDS.m".

3) Run "test_segmentation_BSDS.m" in MATLAB 

2. Experiments #2: MSRC object recognition database (MSRC)

1) Download (MSRC) at http://research.microsoft.com/en-us/projects/objectclassrecognition.
   - Pixel-wise labelled image database v2(591 images, 23 object classes)

2) Set the MSRC directory path to a variable "imgRoot" in "test_segmentation_MSRC.m".

3) Run "test_segmentation_MSRC.m" in MATLAB 

===========================================================================
Note:
===========================================================================
This code uses some graph functions in the Graph Analysis Toolbox (http://eslab.bu.edu/software/graphanalysis/),
the SC display function (http://www.mathworks.com/matlabcentral/fileexchange/16233),
and the EDISON Wrapper for Mean Shift.

The K-means file is additionally included.
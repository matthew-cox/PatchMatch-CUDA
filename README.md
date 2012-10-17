PatchMatch-CUDA
===============

Earl (Bob) Kinney  
Matthew Cox  
CS264  

## Final Project - Patch Match

1. The initial goal of the project was to be able to build a set of tools for 
personal use in modifying images we've personally taken in order to crop
random passer-bys, etc.  

   In practice, the application(s) we have now are simply a demonstration of 
algorithms discussed in the PatchMatch paper and could be used to build such
tools in the future with an additional investment of effort.

   We initially targeted NNF, BDS, image retargeting, and image retargeting
with constraints.  We were only able to get NNF, BDS, and image retargeting
completed.  We have to applications to show off these:
    
    * **reconstruct** - reconstruct an image only using data from another image
    * **retarget** - do a "retarget" resizing of an image
    
2. We used Middlebury Stereoscopic Datasets from 2006:

   > H. Hirschmüller and D. Scharstein. Evaluation of cost functions for stereo matching.
   > In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007), Minneapolis, MN, June 2007. 
   > http://vision.middlebury.edu/stereo/data/scenes2006/

   We also tested with the same Creative Commons images mentioned in the paper. From the following Flickr users: 
    * Sevenbrane (greek_temple_win.bmp)
    * Moi of Ra (flowers_win.bmp)
    * Celie (pagoda_win.bmp)

3. We started by implementing the Nearest Neighbor Field (NNF) code first, as this was a
building block for all other portions of the PatchMatch paper.  We built a 
simple data type (mapent) for storing reference to pixels in an image and relative 
distances in color space.  

   The rest of the code is simply done via procedural programming.  We 
unfortunately did not have enough time to build any sort of OO design for
this project.

4. Both applications (reconstruct and retarget) are command line only 
applications.

   Start by entering the 'src' directory and running 'make'. Then both 
   reconstruct and retarget will be available as per the examples.

   Example of reconstruct:
   
   
        bin/reconstruct ../data/Flower1-view1.bmp ../data/Flower1-view5.bmp
                        ^--The image to recon     ^--The image to pull data from
   
   
   Example of retarget:
   
        
        bin/retarget 3 ../data/Flower1-view1.bmp 500 500
           how fast--^   ^--orig image  new width--^ ^-- new height
          
    
    Options for 'how fast' argument:

    1. **Slow**   - CPU code for NNF and BDS
    2. **Medium** - GPU NNF, CPU BDS
    3. **Fast**   - GPU code for NNF and BDS
  

5. As we implemented both CPU and GPU code for our applications, some of
the speedups may be skewed, but with only minor optimizations (texture 
cache, minimizing data movement, etc) we were able to achieve a 3x speed up 
for reconstruction and 20x speed up for retargeting.

  Example timings below:

  NNF reconstruction:

        Patch size: 11 x 11, Total CPU time:  20847.367 msec
                 Average Iteration CPU time:   4169.474 msec

        Patch size: 11 x 11, Total GPU time:   6739.012 msec
                 Average Iteration GPU time:   1347.802 msec

        Patch size: 11 x 11, Total CPU/CPU time: 167289.922 msec
        Patch size: 11 x 11, Total CPU/GPU time: 55017.648 msec
        Patch size: 11 x 11, Total GPU/GPU time: 2302.331 msec
  
  This is for a single BDS run (with 5 BDS calls and 25 NNF calls) with retarget.

6. A number of improvements enhancements could be made:

   * Better random number generation, particularly in GPU code.  This seems to be the major 
     source of difference in the quality results produced by reconstruct. Try the images of the baby.
   * Importance masks for retarget.  In the current implementation retargeting 
     is attracted to "unique" portions of image.
   * Constraints for retargeting, to maintain straight lines


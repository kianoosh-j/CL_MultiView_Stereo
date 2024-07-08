The program takes a matrix of images, created by a camera array system, as input and produces a depth map for each of them. It consists of three main parts: image segmentation, depth initialization, and depth refinement.

For image segmentation, linear iterative clustering (SLIC) is used to partition each image into super-pixels. Our implementations are in OpenCL, and it is base on the Oxford's gSLICr library, which is written with CUDA.

For depth initialization, we use a simple block matching method to estimate the disparty in two direction of height and width.

For depth refinement, we use a plane-based optimzation technique to iteratively improve smoothness and consistency costs over all super-pixels across images.  
 

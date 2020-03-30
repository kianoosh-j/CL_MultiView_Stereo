This repository contains codes for the project multi-view stereo with OpenCL.

1- Installation:
In order to run this project you need to have microsoft visual studio, OpenCV library, and OpenCL 2.1 installed and integerated.

2- Describtion:
This program takes a matrix of images, created by a camera array system, as input and produces a depth map for each of them. It consists of three main parts: image segmentation, depth initialization, and depth refinement.

For image segmentation, we use simple linear iterative clustering (SLIC) to partition each image into super-pixels. Our OpenCL implementation of SLIC is base on the Oxford's gSLICr library which is written with CUDA.

For depth initialization, we use a simple block matching method to estimate the disparty in two direction of height and width.

For depth refinement, we use a complex plane-based optimzation technique to iteratively improve smoothness and consistency costs over all super-pixels for all of the images.

3- results:
Results of each stage of the program for dataset have been uploaded.   
 

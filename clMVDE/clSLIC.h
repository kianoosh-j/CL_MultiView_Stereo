#pragma once
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include "header.h"

//#define LOCAL_SIZE 16

class clSLIC
{

public: 
	
	clSLIC(cl::Program program, system_settings *settings_, cl_int2 img_size, cl_int2 map_size);
	~clSLIC();


	void do_super_pixel_seg(vec3u *in_img, vec3f *cvt_img, vec8f *spixl_map, cl_uint *idx_img);
	void draw_segmentation_lines(cl_uchar3 *in_img, cl_uchar3 *out_img);

private:

	system_settings *settings;

	// Variables
	float max_color_dist;
	float max_xy_dist;
	int num_grid_per_center;
	
	vec2i img_size;
	vec2i map_size;

	vec3u *in_img;
	vec8f *accum_map;
	vec3f *lab_img;
	vec8f *spixl_map;
	cl_uint *idx_img;
	
	cl::Buffer *lab_img_dev;
	cl::Buffer *in_img_dev;
	cl::Buffer *spixl_map_dev;
	cl::Buffer *idx_img_dev;
	cl::Buffer *edge_val_dev;
	//cl::Buffer *accum_map_dev;
	cl::Program program;

	// Methods
	void cvt_color_space();
	void init_cluster_centers();
	void find_center_association();
	void update_cluster_center();
	void enforce_connectivity();
	void alternative_update_cluster();
	void apply_edge_values();
	void init_labels();
};

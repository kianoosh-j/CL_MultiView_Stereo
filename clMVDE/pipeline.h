#include "header.h"
#include "clSLIC.h"
#include "photo_consistency.h"
#include "depth_refinement.h"

class pipeline
{

public:
	pipeline(std::string file_addr, system_settings *settings);
	void exe_pipeline();

	// Variables:
	cl::Program program;
	vec2i img_size, map_size;
	system_settings *settings;
	int view_num;

private:

	void perform_segmentation();
	void perform_depth_est();
	void init_depth_map(std::vector<std::vector<int> > &view_subset, std::vector<float> &disp_levels, int num_disp_levels);
	void refine_depth_map(std::vector<std::vector<int> > &view_subset);
	void compile_device_code();


	// Arrays
	vector<Mat> img_array;
	vec3f *cvt_img;
	cl_uint *idx_img;
	vec8f *spixl_map;
	vec8u *spixl_rep;
	//cl::Buffer *cvt_img, *idx_img, *spixl_map;

};
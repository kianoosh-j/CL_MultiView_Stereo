#include "header.h"



class clPhotoConsistency
{
public:
	clPhotoConsistency(cl::Program program, int view_count, int spixl_size, int num_disp_levels, vec2i img_size_, vec2i map_size_);

	void do_initial_depth_estimation(
		vec8f *spixel_map,
		vec8u *spixel_rep,
		vec3f *cvg_img,
		cl_uint   *idx_img,
		int array_width,
		float bl_ratio,
		vector<vector<int> > &view_subset,
		vector<float> &disp_levels
	);

	~clPhotoConsistency();

	void do_initial_depth_estimation_host(
		vec8f *spixel_map,
		vec8u *spixel_rep,
		vec3f *in_img,
		cl_uint   *idx_img,
		int array_width,
		float bl_ratio,
		vector<float> &disp_levels,
		int *view_subset_mat,
		int *subset_num_per_cam,
		int start_view, int end_view
	);

	//void find_super_pixel_boundary(vec8f *spixl_map, cl_uint *idx_img, )

private:
	
	//variables
	cl::Program program;
	int spixel_size;
	int num_disp_levels;
	int view_count;
	int no_disp;

	vec2i map_size;
	vec2i img_size;

	// functions
	void init_depth_map(vec8f *spixl_map, vec8u *spixl_rep, vec3f *cvt_img, cl_uint *idx_img, int *view_subset_num, int *view_subset_vec, float *disp_levels, int x, int y, int z);
	void compare_host_to_device(vec8f *a_d, vec8f *a_h, int start_view, int end_view);
	void img_translate(cl_uint *idx_img, cl_float8 *spixel_map, float min_disp, float max_disp, int start_view, int end_view);
};
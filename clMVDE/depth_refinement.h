#include "header.h"

class clDepthRefinement
{
public:
	clDepthRefinement(cl::Program program, vec2i img_size, vec2i map_size, vec2i camera_array_size, vec3f *cvt_img, vec8f *spixl_map, 
		cl_uint *idx_img, vec8u *spixl_rep, std::vector<std::vector<int> > view_subset_vec, int spixl_size, float bl_ratio);
	~clDepthRefinement();
	void do_refinement(float gamma, float alpha, float fuse, int kernel_step, int kernel_size, int no_prop);

private:
	// Variables
	vec2i img_size;
	vec2i map_size;
	int array_width;
	int view_count;
	int spixl_size;
	float bl_ratio;
	cl::Program program;

	// Host Arrays
	int *subset_num;
	int *view_subset;
	vec2f *flatness_map;
	float *current_state;
	float *current_state2;
	float *current_state_host; // For Test
	float *current_state_host2;	// For Test
	vec3f *cvt_img;
	vec8f *spixl_map;
	cl_uint *idx_img;
	vec8u *spixl_rep;
	float *disp_img;
	
	
	// Device Buffers
	cl::Buffer *cvt_img_dev;
	cl::Buffer *spixl_map_dev;
	cl::Buffer *idx_img_dev;
	cl::Buffer *spixl_rep_dev;
	cl::Buffer *flatness_map_dev;
	cl::Buffer *view_subset_dev;
	cl::Buffer *subset_num_dev;
	cl::Buffer *current_state_dev;
	cl::Buffer *current_state_dev2;
	cl::Buffer *disp_full_dev;
	cl::Buffer *disp_proj_dev;
	cl::Buffer *disp_img_dev;
	

	// Methods:
	void compute_flatness(float gamma);
	void compute_flatness_host(vec2f *flatness_map_host, float gamma);
	void compute_flatness_for_spixl(vec2f *flatness_map_host, float gamma, int x, int y, int z);
	void compare_host_to_device_flatness(vec2f *a_h, vec2f *a_d, int start_view, int end_view);
	void img_translate_flatness(cl_float2 *flatness, int start_view, int end_view);


	void init_spixl_state(float gamma_, float alpha_, int kernel_steps_, int kernel_size_, float fuse_);
	void init_spixl_state_host(float *current_state_host, int start_view, int end_view, float gamma, float alpha, float fuse, int no_kernel_steps, float kernel_step_size);
	void init_state_per_element(float *current_state_host, int x, int y, int z, float gamma, float alpha, float fuse, int no_kernel_steps, int kernel_step_size);
	float init_smoothness(vec8f *spixl_map, vec8f sp_ref, vec2f fl, vec2i map_size, vec3i pos, float gamma, float alpha, int no_kernel_steps, float kernel_step_size);
	float init_consistency(int *view_subset, int *subset_num, vec3i pos, int array_width, float bl_ratio, float fuse, float alpha, float gamma, vec2f fl);
	void compare_host_to_device_state(float *a_h, float * a_d, int start_view, int end_view, int element_num);
	void img_translate_state(float *current_state, int start_view, int end_view, int element_num, int min_disp, int max_disp, std::string file_name);


	void propagate(int iter, int norm, float gamma_, float alpha_, int kernel_steps_, int kernel_size_, float fuse_);
	void propagate_host(float *current_state_host, float *current_state_host2, int iter, int norm, float gamma, float alpha, int no_kernel_steps, float kernel_step_size, float fuse, int start_view, int end_view);
	void propagate_current_thread(float *current_state, float *current_state_update, int iter, int norm, float gamma, float alpha, int no_kernel_steps, float kernel_step_size, float fuse, int x, int y, int z);
	vec8f update_current_thread(float *current_state_host, int idx_check, int idx_state_check, vec2f center, vec3f color, int iter, vec3f n0, float sm0, float cs0, float d0, float alpha, float gamma, float fuse, float bl_ratio, vec2f fl, int x, int y, int z, int no_kernel_steps, float kernel_step_size);
	vec8f refine_current_thread(float *current_state_host, vec2i nbr1, vec2i nbr2, int x, int y, int z, vec2f center, vec3f color, float d0, float sm0, float cs0, vec3f n0, vec2f fl, float alpha, float gamma, float bl_ratio, float fuse, int no_kernel_steps, float kernel_step_size, int iter);
	float compute_smoothness(float *current_state_host, vec3f color, vec2f center, float d, vec3f n, int x, int y, int z, float alpha, float gamma, int no_kernel_steps, float kernel_step_size, vec2f fl);
	float compute_consistency(float *current_state_host, vec3f color, vec2f center, int x, int y, int z, float d, vec3f n, float alpha, float gamma, float bl_ratio, float fuse, vec2f fl);


	void fusion(float alpha_, float gamma_, float bl_ratio_, float fuse_);
	void plot_full_image(int start_view, int end_view, int min_disp, int max_disp, std::string file_name);


};
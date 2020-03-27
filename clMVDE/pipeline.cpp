#include "stdafx.h"

#include "pipeline.h"
#include "file_handler.h"


pipeline::pipeline(std::string file_addr, system_settings *settings_)
{
	this->settings = settings_;
	view_num = settings->array_height * settings->array_width;
	
	if (read_image_array(img_array, file_addr, view_num))
		std::cout << " error read input files !!" << std::endl;
	
	img_size.x  = img_array[0].size().width;
	img_size.y = img_array[0].size().height;

	map_size.x  = (int)ceil((float)img_size.x / (float)settings->spixl_size);
	map_size.y = (int)ceil((float)img_size.y / (float)settings->spixl_size);

	// Compile Device Code 
	compile_device_code();

	// Allocating Memory Arrays
	cvt_img   = new vec3f[img_size.x * img_size.y * view_num];
	idx_img   = new cl_uint[img_size.x * img_size.y * view_num];
	spixl_map = new vec8f[map_size.x * map_size.y * view_num];
}


void pipeline::compile_device_code()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	auto default_platform = platforms.begin();
	std::vector<cl::Device> devices;
	default_platform->getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Context context(devices);

	std::ifstream kernel_file("clcode.cl");
	std::string src((std::istreambuf_iterator<char>(kernel_file)),
		std::istreambuf_iterator<char>());
	const char* src_new = src.c_str();
	cl::Program program_temp(context, cl::Program::Sources(
		1, std::make_pair(src_new, src.length() + 1)
	));

	this->program = program_temp;

	cl_int err = program.build(devices);

	if (err != CL_SUCCESS)
		std::cout << "Build Error. Error Number: " << err << std::endl;
}



void pipeline::exe_pipeline()
{
	perform_segmentation();
	//perform_depth_est();
}



void pipeline::perform_segmentation()
{
	vec3u *in_img  = new vec3u[img_size.x * img_size.y * view_num];
	vec3u *res_img = new vec3u[img_size.x * img_size.y * view_num];
	
	clSLIC *slic_obj = new clSLIC(program, this->settings, img_size, map_size);
	
	//namedWindow("MyWindow");
	for (int i = 0 ; i < view_num ; i++)
	{
		auto in_img_start	 = in_img	 + i*img_size.x * img_size.y;
		auto idx_img_start	 = idx_img	 + i*img_size.x * img_size.y;
		auto cvt_img_start	 = cvt_img   + i*img_size.x * img_size.y;
		auto spixl_map_start = spixl_map + i*map_size.x * map_size.y;
		auto res_img_start	 = res_img	 + i*img_size.x * img_size.y;

		Mat pic = img_array[i];
		loadImageIn(pic, in_img_start, img_size.y, img_size.x);

		auto start_time = std::chrono::high_resolution_clock::now();	// Take the start time
		slic_obj->do_super_pixel_seg(in_img_start, cvt_img_start, spixl_map_start, idx_img_start);
		auto end_time = std::chrono::high_resolution_clock::now();	// Take the end time

		auto duration = duration_cast<milliseconds>(end_time - start_time);
		std::cout << "Time of SLIC = " << duration.count() << std::endl;

		//slic_obj->draw_segmentation_lines(in_img_start, res_img_start);
	}

	/**/
	delete slic_obj;
	delete in_img;
	delete res_img;
	/**/
	// Show result
	//show_img(res_img, img_size.y, img_size.x, view_num);
}



void pipeline::perform_depth_est()
{
	int neib_hor = settings->neib_hor;
	int neib_ver = settings->neib_ver;
	int min_disp = settings->min_disp;
	int max_disp = settings->max_disp;
	int inc = settings->inc;
	int num_disp_levels = settings->num_disp_levels;


	/////////////////////////////////////////////////////////
	///////////////// Set Disparity Levels /////////////////
	/////////////////////////////////////////////////////////
	std::vector<float> disp_levels;
	for (int i = 0 ; i <= (max_disp - min_disp) / inc ; i++)
		disp_levels.push_back(min_disp + i*inc);
	num_disp_levels = disp_levels.size();


	/////////////////////////////////////////////////////////////
	///////////////// Set Camera View Matrices /////////////////
	///////////////////////////////////////////////////////////
	vector<vector<int> > view_subset;
	view_subset.resize(view_num);

	for (int i = 0 ; i < view_num ; i++)
	{
		for (int x = i % settings->array_width - neib_hor ; x <= i % settings->array_width + neib_hor ; x++)
			for (int y = i / settings->array_width - neib_ver ; y <= i / settings->array_width + neib_ver ; y++)
			{
				int indx = y*settings->array_width + x;
				if (x >= 0 && x < settings->array_width && y >= 0 && y < settings->array_height &&  indx != i)
					view_subset[i].push_back(indx);
			}
	}

	/////////////////////////////////////////////////////////
	///////////////// Run Depth Estimation /////////////////
	///////////////////////////////////////////////////////
	spixl_rep = new vec8u[map_size.x * map_size.y * view_num];
	
	init_depth_map(view_subset, disp_levels, num_disp_levels);
	refine_depth_map(view_subset);
}


void pipeline::init_depth_map(std::vector<std::vector<int> > &view_subset, std::vector<float> &disp_levels, int num_disp_levels)
{
	clPhotoConsistency *consistency_obj = new clPhotoConsistency(program, view_num, settings->spixl_size, num_disp_levels, img_size, map_size);
	consistency_obj->do_initial_depth_estimation(spixl_map, spixl_rep, cvt_img, idx_img, settings->array_width, settings->bl_ratio, view_subset, disp_levels);
	//delete consistency_obj;
}


void pipeline::refine_depth_map(std::vector<std::vector<int> > &view_subset)
{
	float gamma = 2 * pow(settings->gamma, 2);
	float alpha = 2 * pow(settings->alpha, 2);
	int kernel_size = settings->kernel_size / 2;
	vec2i camera_array_size = makeVec2i(settings->array_width, settings->array_height);
	//camera_array_size.x = settings->array_width; camera_array_size.y = settings->array_height;

	clDepthRefinement *refinement_obj = new clDepthRefinement(program, img_size, map_size, camera_array_size, cvt_img, 
																		spixl_map, idx_img, spixl_rep, view_subset, settings->spixl_size, settings->bl_ratio);

	refinement_obj->do_refinement(gamma, alpha, settings->fuse, settings->kernel_step, kernel_size, settings->no_prop);
	//delete refinement_obj;
}


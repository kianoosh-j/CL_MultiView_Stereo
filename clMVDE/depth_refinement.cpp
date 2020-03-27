#include "stdafx.h"
#include "depth_refinement.h"


clDepthRefinement::clDepthRefinement(cl::Program program, vec2i img_size, vec2i map_size, vec2i camera_array_size, vec3f *cvt_img, vec8f *spixl_map,
	 cl_uint *idx_img, vec8u *spixl_rep, std::vector<std::vector<int> > view_subset_vec, int spixl_size, float bl_ratio)
{
	this->program  = program;
	this->img_size = img_size;
	this->map_size = map_size;
	this->spixl_size = spixl_size;
	this->bl_ratio  = bl_ratio;
	this->array_width = camera_array_size.x;
	this->view_count = camera_array_size.x * camera_array_size.y;

	this->cvt_img   = cvt_img;
	this->spixl_map = spixl_map;
	this->idx_img   = idx_img;
	this->spixl_rep = spixl_rep;
	

	// 
	subset_num  = new int[view_count];
	view_subset = new int[view_count * view_count];

	for (int i = 0 ; i < view_count ; i++)
	{
		subset_num[i] = view_subset_vec[i].size();

		for (int j = 0 ; j < view_subset_vec[i].size() ; j++)
			view_subset[i * view_count + j] = view_subset_vec[i][j];
	}

	// Allocate Buffers
	cl_int err;

	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();


	view_subset_dev = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		view_count*view_count * sizeof(int), view_subset, &err);

	subset_num_dev = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		view_count * sizeof(int), subset_num, &err);

	//delete view_subset; 
	//delete subset_num;

	cvt_img_dev = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		img_size.x * img_size.y * view_count * sizeof(vec3f), cvt_img, &err);

	spixl_map_dev = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		map_size.x * map_size.y * view_count * sizeof(vec8f), spixl_map, &err);

	idx_img_dev = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		img_size.x * img_size.y * view_count * sizeof(cl_uint), idx_img, &err);

	spixl_rep_dev  = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		map_size.x * map_size.y * view_count * sizeof(vec8u), spixl_rep, &err);

	flatness_map_dev = new cl::Buffer(context, CL_MEM_READ_WRITE,
		map_size.x * map_size.y * view_count * sizeof(vec2f), nullptr, &err);

	current_state_dev = new cl::Buffer(context, CL_MEM_READ_WRITE,
		6 * map_size.x * map_size.y * view_count * sizeof(float), nullptr, &err);

	current_state_dev2 = new cl::Buffer(context, CL_MEM_READ_WRITE,
		6 * map_size.x * map_size.y * view_count * sizeof(float), nullptr, &err);

}

clDepthRefinement::~clDepthRefinement()
{
	delete view_subset;
	delete subset_num;
	delete current_state;
	delete current_state2;


	delete spixl_map_dev;
	delete idx_img_dev;
	delete cvt_img_dev;
	delete spixl_rep_dev;
	delete current_state_dev;
	delete current_state_dev2;

}


void clDepthRefinement::do_refinement(float gamma, float alpha, float fuse, int kernel_step, int kernel_size, int no_prop)
{
	compute_flatness(gamma);
	
	// for test //
	this->current_state = new float[6 * map_size.x * map_size.y * view_count];
	this->current_state2 = new float[6 * map_size.x * map_size.y * view_count];

	this->current_state_host = new float[6 * map_size.x * map_size.y * view_count];
	this->current_state_host2 = new float[6 * map_size.x * map_size.y * view_count];
	///////////////

	init_spixl_state(gamma, alpha, kernel_step, kernel_size, fuse);

	for (int i = 0 ; i < no_prop ; i++)
		propagate(i, i > -1, gamma, alpha, kernel_step, kernel_size, fuse); // Ask Question hheeeeeeeeeeeeeeeeeeeeeeeeeerrrrrrrrrrrrrreeeeeeeeee

	// -------------------------------------------
	// Fusion: Project back to Image
	fusion(alpha, gamma, bl_ratio, fuse);

	// for test //
	delete current_state;
	delete current_state2;
	delete current_state_host;
	delete current_state_host2;
	/////////
}



void clDepthRefinement::compute_flatness(float gamma_)
{
	std::cout << "Compute Flatness" << std::endl;

	cl_int err;
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	float gamma = (1.0 / (float) gamma_);
	// Kernel Definition
	cl::Kernel kernel(program, "compute_flatness");
	err = kernel.setArg(0, *spixl_map_dev);
	err = kernel.setArg(1, *flatness_map_dev);
	err = kernel.setArg(2, map_size);
	err = kernel.setArg(3, gamma);

	if (err != CL_SUCCESS)	std::cout << " Flatness Set Arguman Error. Error Number: " << err << std::endl;

	// Kernel Execution Configuration
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float)map_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)map_size.y / LOCAL_SIZE)) * LOCAL_SIZE;

	// Execute the Kernel on Device
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y, view_count), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE, 1));
	if (err != CL_SUCCESS)
		std::cout << " Flatness Kernel Launch Error. Error Number: " << err << std::endl;



	// Testing //
	flatness_map = new vec2f[map_size.x * map_size.y * view_count];

	err = queue.enqueueReadBuffer(*flatness_map_dev, CL_TRUE, 0, map_size.x * map_size.y * view_count * sizeof(vec2f), flatness_map);

	if (err != CL_SUCCESS)	std::cout << " Flatness Read Buffer Error. Error Number: " << err << std::endl;
	//img_translate_flatness(flatness_map, 0, view_count);

	/**
	vec2f *flatness_map_host = new vec2f[map_size.x * map_size.y * view_count];

	compute_flatness_host(flatness_map_host, gamma);
	compare_host_to_device_flatness(flatness_map_host, flatness_map, 0, view_count);
	//img_translate_flatness(flatness_map_host, 0, view_count);
	
	/**

	printText("Flatness Host: ");
	for (int i = 20 ; i < 25 ; i++)
	{
		for (int j = 20 ; j < 25 ; j++)
		{
			std::cout << flatness_map_host[(map_size.x * map_size.y * 0) + (map_size.x * i) + j].x << ", ";
		}
		std::cout << std::endl;
	}

	printText("Flatness Device: ");
	for (int i = 20 ; i < 25 ; i++)
	{
		for (int j = 20 ; j < 25 ; j++)
		{
			std::cout << flatness_map[(map_size.x * map_size.y * 0) + (map_size.x * i) + j].x << ", ";
		}
		std::cout << std::endl;
	}

	// end test //

	/**/
}



void clDepthRefinement::compare_host_to_device_flatness(vec2f *a_h, vec2f *a_d, int start_view, int end_view)
{
	printText("Compare Flatness Host and Device:");
	int miss = 0;
	int zero_count = 0, non_zero_count = 0;

	for (int z = start_view ; z < end_view ; z++)
		for (int y = 0 ; y < map_size.y ; y++)
			for (int x = 0 ; x < map_size.x ; x++)
			{
				int idx = z * map_size.x * map_size.y + y * map_size.x + x;
					
				if (abs(a_d[idx].s1 - a_h[idx].s1) > 0.0001)
				{
					miss++;
					if (miss < 15)
						std::cout << "x = " << x << " y = " << y << ", a_d[" << idx << "] = " << a_d[idx].s0 << ", a_h[" << idx << "] = " << a_h[idx].s0 << std::endl;
				}

				if (abs(a_d[idx].s0 - a_h[idx].s0) > 0.0001)
					miss++;

				if (a_d[idx].s0 == 0.0)
					zero_count++;
				else
					non_zero_count++;
			}

	std::cout << "miss = " << miss << std::endl;
	std::cout << "zero_count percentage = " << ((float)zero_count / (float)(non_zero_count + zero_count)) * 100 << std::endl;

}



void clDepthRefinement::compute_flatness_for_spixl(vec2f *flatness_map_host, float gamma, int x, int y, int z)
{
	int idx = z*map_size.x*map_size.y + y*map_size.x + x;

	vec3f c1;
	vec3f c0;

	c0.x = spixl_map[idx].s3;
	c0.y = spixl_map[idx].s4;
	c0.z = spixl_map[idx].s5;

	float diff = 0;
	float fl = 1.0;

	if (x - 1 >= 0)
	{
		c1.x = spixl_map[idx - 1].s3;
		c1.y = spixl_map[idx - 1].s4;
		c1.z = spixl_map[idx - 1].s5;
		diff = (c1.x - c0.x)*(c1.x - c0.x) + (c1.y - c0.y)*(c1.y - c0.y) + (c1.z - c0.z)*(c1.z - c0.z);
		fl += diff;
	}
	
	if (x + 1 < map_size.x)
	{
		c1.x = spixl_map[idx + 1].s3;
		c1.y = spixl_map[idx + 1].s4;
		c1.z = spixl_map[idx + 1].s5;
		diff = (c1.x - c0.x)*(c1.x - c0.x) + (c1.y - c0.y)*(c1.y - c0.y) + (c1.z - c0.z)*(c1.z - c0.z);
		fl += diff;
	}
	
	if (y - 1 >= 0)
	{
		c1.x = spixl_map[idx - map_size.x].s3;
		c1.y = spixl_map[idx - map_size.x].s4;
		c1.z = spixl_map[idx - map_size.x].s5;
		diff = (c1.x - c0.x)*(c1.x - c0.x) + (c1.y - c0.y)*(c1.y - c0.y) + (c1.z - c0.z)*(c1.z - c0.z);
		fl += diff;
	}
	
	if (y + 1 < map_size.y)
	{
		c1.x = spixl_map[idx + map_size.x].s3;
		c1.y = spixl_map[idx + map_size.x].s4;
		c1.z = spixl_map[idx + map_size.x].s5;
		diff = (c1.x - c0.x)*(c1.x - c0.x) + (c1.y - c0.y)*(c1.y - c0.y) + (c1.z - c0.z)*(c1.z - c0.z);
		fl += diff;
	}
	
	flatness_map_host[idx].s0 = exp(-fl*gamma);
	flatness_map_host[idx].s1 = 1 - exp(-0.25*fl*gamma);
}



void clDepthRefinement::compute_flatness_host(vec2f *flatness_map_host, float gamma)
{
	for (int z = 0 ; z < view_count ; z++)
		for (int y = 0 ; y < map_size.y ; y++)
			for (int x = 0; x < map_size.x; x++)
				compute_flatness_for_spixl(flatness_map_host, gamma, x, y, z);

}



void clDepthRefinement::img_translate_flatness(cl_float2 *flatness, int start_view, int end_view)
{
	Mat test_cam(img_size.y, img_size.x, CV_32FC1);
	Mat test_cam_2(img_size.y, img_size.x, CV_8UC1);
	//namedWindow("My Window", WINDOW_AUTOSIZE);

	for (int k = start_view ; k < end_view ; k++)
	{
		for (int i = 0 ; i < img_size.y ; i++)
			for (int j = 0 ; j < img_size.x ; j++)
			{
				int sp_idx = idx_img[img_size.s0*img_size.s1*k + img_size.s0*i + j];
				float d = flatness[map_size.s0*map_size.s1*k + sp_idx].s0;
				
				unsigned char d_scale = (unsigned char)ceil(d * 255);
				test_cam.at<float>(i, j) = d;
				test_cam_2.at<unsigned char>(i, j) = d_scale;
			}
		std::string address = "../results/2- flatness/flatness_new " + std::to_string(k) + ".png";
		imwrite(address, test_cam_2);
		//imshow("My Window", test_cam);
		//waitKey(0);
	}
}




void clDepthRefinement::init_spixl_state(float gamma_, float alpha_, int kernel_steps_, int kernel_size_, float fuse_)
{

	std::cout << "Initialize State of the Map" << std::endl;
	
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	float gamma = 1 / gamma_;
	float alpha = 1 / alpha_;
	int kernel_steps = kernel_steps_;
	float sp_kernel_step = std::max(1, kernel_size_ / kernel_steps*spixl_size);
	float fuse = 0.5*fuse_;
	cl_int err;

	// Define Kernel 
	cl::Kernel kernel(program, "init_current_state");
	err = kernel.setArg(0, *spixl_map_dev);
	err = kernel.setArg(1, *idx_img_dev);
	err = kernel.setArg(2, *spixl_rep_dev);
	err = kernel.setArg(3, *flatness_map_dev);
	err = kernel.setArg(4, *current_state_dev);
	err = kernel.setArg(5, gamma);
	err = kernel.setArg(6, alpha);
	err = kernel.setArg(7, kernel_steps);
	err = kernel.setArg(8, sp_kernel_step);
	err = kernel.setArg(9, bl_ratio);
	err = kernel.setArg(10, fuse);
	err = kernel.setArg(11, map_size);
	err = kernel.setArg(12, *view_subset_dev);
	err = kernel.setArg(13, *subset_num_dev);
	err = kernel.setArg(14, array_width);
	err = kernel.setArg(15, img_size);
	err = kernel.setArg(16, view_count);
	if (err != CL_SUCCESS)	std::cout << " Init_State. Set Arguman Error. Error no: " << err << std::endl;


	// Configure Kernel 
	cl_int3 grid_size;
	grid_size.x = (int)(ceil((float) map_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float) map_size.y / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.z = view_count;

	// Execute the Kernel
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y, grid_size.z), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE, 1));
	if (err != CL_SUCCESS)	std::cout << " Init_State. Kernel Execution Error. Error no: " << err << std::endl;


	/////////////////////////////////////
	//////////////// Test //////////////
	///////////////////////////////////

	//-------------------------------------------------------------------------------------------------------------------
	/**/
	this->current_state	  = new float[6 * map_size.x * map_size.y * view_count];

	// Transfer from Host to Device
	err = queue.enqueueReadBuffer(*current_state_dev, CL_TRUE, 0, 6 * map_size.x * map_size.y * view_count * sizeof(cl_float), this->current_state); queue.finish();
	if (err != CL_SUCCESS)	std::cout << " Init_State: Read Buffer Error. Error Number: " << err << std::endl;

	//------------------------------------------------------------------------------------------------------------------
	/**
	img_translate_state(current_state, 0, view_count, 1, 10, 100, "3- initialize smoothness/initSm_dev");
	img_translate_state(current_state, 0, view_count, 2, 10, 100, "4- initialize consistency/initCs_dev");
	/**/

	//-----------------------------------------------------------------------------------------------------------------
	/**
	init_spixl_state_host(this->current_state_host, 0, view_count, gamma, alpha, fuse, kernel_steps, sp_kernel_step);
	compare_host_to_device_state(current_state_host, this->current_state, 0, view_count, 1);
	//img_translate_state(current_state, 0, view_count, 1, 10, 100, "init_sm_dev");
	/**/
	
}



void clDepthRefinement::compare_host_to_device_state(float *a_h, float * a_d, int start_view, int end_view, int element_num)
{
	printText("Compare State Host to Device: ");

	switch (element_num)
	{
	case 0: printText("Depth Comparison:");
		break;
	case 1: printText("Smoothness Comparison:");
		break;
	case 2: printText("Consistency Comparison:");
		break;
	case 3: printText("Norm X Comparison:");
		break;
	case 4: printText("Norm Y Comparison:");
		break;
	case 5: printText("Norm Z Comparison:");
		break;
	}

	int miss = 0;
	int zero_count = 0, non_zero_count = 0;

	for (int z = start_view ; z < end_view ; z++)
		for (int y = 0 ; y < map_size.y ; y++)
			for (int x = 0 ; x < map_size.x ; x++)
			{
				int idx = (6 * map_size.x * map_size.y * z) + (6 * map_size.x * y) + (6 * x);

				if (abs(a_d[idx + element_num] - a_h[idx + element_num]) > 0.001)
				{
					miss++;
					if (miss < 10)
						std::cout << "x = " << x << " y = " << y << ", a_d[" << idx << "] = " << a_d[idx + element_num] << ", a_h[" << idx << "] = " << a_h[idx + element_num] << std::endl;
				}

				if (a_d[idx + element_num] == 0.0)
					zero_count++;
				else
					non_zero_count++;

			}

	std::cout << "miss = " << miss << std::endl;
	std::cout << "zero_count percentage = " << ((float)zero_count / (float)(non_zero_count + zero_count)) * 100 << std::endl;

}



float clDepthRefinement::init_consistency(int *view_subset, int *subset_num, vec3i pos, int array_width, float bl_ratio, float fuse, float alpha, float gamma, vec2f fl)
{
	vec8f spixl_ref = spixl_map[pos.z*map_size.x*map_size.y + map_size.x*pos.y + pos.x];

	// Load SuperPixel Info
	vec3f color;  
	color.x = spixl_ref.s3; 
	color.y = spixl_ref.s4; 
	color.z = spixl_ref.s5;

	vec2f center; 
	center.x = spixl_ref.s1; 
	center.y = spixl_ref.s2;
	
	float d = spixl_ref.s7;	// SuperPixel Dispartiy

	// Other Parameters
	float consistency = 0.0;

	int view_counter = 0;
	vec2i camIdx;
	camIdx.x = pos.z % array_width;
	camIdx.y = pos.z / array_width;

	// Super pixels Samples
	vec8u rep = spixl_rep[pos.z*map_size.x*map_size.y + pos.y*map_size.x + pos.x];

	int sp_samples[9];
	sp_samples[0] = (int)rep.s0;
	sp_samples[1] = (int)rep.s1;
	sp_samples[2] = (int)rep.s2;
	sp_samples[3] = (int)rep.s3;
	sp_samples[4] = 0;
	sp_samples[5] = (int)rep.s4;
	sp_samples[6] = (int)rep.s5;
	sp_samples[7] = (int)rep.s6;
	sp_samples[8] = (int)rep.s7;

	// Main For Loop
	for (int n = 0 ; n < subset_num[pos.z] ; n++)
	{
		
		int view = view_subset[pos.z*view_count + n];

		vec2i viewIdx;
		viewIdx.x = view % array_width;
		viewIdx.y = view / array_width;

		// Consistency Variables
		float visib_weight_sum = 0.0f;
		float occl_weight_sum = 0.0;
		float num = 0.0;
		float visibility = 0.0;
		float visible = 0.0;


		for (int i = -1 ; i <= 1 ; i++) for (int j = -1 ; j <= 1 ; j++)
		{
			vec2i xy_ref;
			xy_ref.x = (int)center.x + sp_samples[(i + 1) * 3 + j + 1] * i;
			xy_ref.y = (int)center.y + sp_samples[(i + 1) * 3 + j + 1] * j;

			vec2i xy_proj;
			xy_proj.x = xy_ref.x - round(d*(viewIdx.x - camIdx.x));
			xy_proj.y = xy_ref.y - round(bl_ratio*d*(viewIdx.y - camIdx.y));

			if (xy_proj.x >= 0 && xy_proj.y >= 0 && xy_proj.x < img_size.x  && xy_proj.y < img_size.y)
			{
				uint idx_proj = idx_img[img_size.x*img_size.y*view + img_size.x*xy_proj.y + xy_proj.x];
				vec2i coord_proj;
				coord_proj.x = idx_proj % map_size.x;
				coord_proj.y = idx_proj / map_size.x;
				vec8f sp_proj = spixl_map[map_size.x*map_size.y*view + map_size.x*coord_proj.y + coord_proj.x];

				float diff = sp_proj.s7 - d;
				float when_visible = 0.0;
				if (abs(diff) < fuse)  when_visible = 1.0;
				visible += when_visible*exp(-diff*diff*alpha);
				visib_weight_sum += when_visible;
				occl_weight_sum += (1 - when_visible);

				vec3f color_proj; 
				color_proj.x = sp_proj.s3; 
				color_proj.y = sp_proj.s4;
				color_proj.z = sp_proj.s5;
				diff = sqrt(pow(color_proj.x - color.x, 2) + pow(color_proj.y - color.y, 2) + pow(color_proj.z - color.z, 2) );
				visibility += exp(-diff * diff * gamma);

				num += 1;
			}
		}

		if (num > 0)
		{
			view_counter++;
			if (visib_weight_sum > 0)
				consistency += (visib_weight_sum / num)*(visibility / visib_weight_sum)*(visible / visib_weight_sum);

			if (occl_weight_sum > 0)
				consistency += 0.5*fl.y;
		}
	}

	if (view_counter > 0)
		return consistency / view_counter;
	else
		return 0.01;
}



float clDepthRefinement::init_smoothness(vec8f *spixl_map, vec8f sp_ref, vec2f fl, vec2i map_size, vec3i pos, float gamma, float alpha, int no_kernel_steps, float kernel_step_size)
{
	float smoothness  = 0.0;
	float weight_norm = 0.0;

	vec3f color = makeVec3f(sp_ref.s3, sp_ref.s4, sp_ref.s5); 
	//color.x = sp_ref.s3;  color.y = sp_ref.s4, color.z = sp_ref.s5;
	float disp = sp_ref.s7;

	// Check the Immidate Neighbors
	for (int i = -1 ; i <= 1 ; i++) for (int j = -1 ; j <= 1 ; j++)
	{
		vec3i pos_check;
		pos_check.x = pos.x + i; 
		pos_check.y = pos.y + j; 
		pos_check.z = pos.z;

		if (pos_check.x >= 0 && pos_check.y >= 0 && pos_check.x < map_size.x && pos_check.y < map_size.y && (i != 0 || j != 0))
		{
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x*map_size.y*pos_check.z + map_size.x*pos_check.y + pos_check.x];
			vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);
			float disp_check = sp_check.s7;

			diff = euDistance3D(color_check, color);// sqrt(pow(color_check.x - color.x, 2) + pow(color_check.y - color.y, 2) + pow(color_check.z - color.z, 2));
			similarity = exp(-diff*diff*gamma);
			diff = disp - disp_check;
			smoothness += similarity * exp(-diff*diff*alpha);
			weight_norm += similarity;
		}		
	}

	int step_size = max(1, (int)(fl.x * kernel_step_size + 0.5));


	// Propagation Kernel 
	for (int i = 1 ; i <= no_kernel_steps ; i++)
	{
		float gamma_i = gamma*(1 + i);
		int step = i*step_size;

		if (pos.x > step) // Left
		{
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x*map_size.y*pos.z + map_size.x*pos.y + pos.x - (step + 1)];
			vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);
			float disp_check = sp_check.s7;

			diff = euDistance3D(color_check, color);
			similarity = exp(-diff * diff * gamma_i);
			diff = disp - disp_check;

			smoothness += similarity * exp(-diff * diff * alpha);
			weight_norm += similarity;
		}

		if (pos.x < map_size.x - step - 1) // Right
		{
			
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x*map_size.y*pos.z + map_size.x*pos.y + pos.x + (step + 1)];
			vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);
			float disp_check = sp_check.s7;

			diff = euDistance3D(color_check, color);
			similarity = exp(-diff*diff*gamma_i);
			diff = disp - disp_check;

			smoothness += similarity * exp(-diff*diff*alpha);
			weight_norm += similarity;
			
		}

		if (pos.y > step)// UP
		{
			
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x * map_size.y * pos.z + map_size.x * (pos.y - step -1) + pos.x];
			vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);
			float disp_check = sp_check.s7;

			diff = euDistance3D(color_check, color);
			similarity = exp(-diff*diff*gamma_i);
			diff = disp - disp_check;

			smoothness += similarity * exp(-diff*diff*alpha);
			weight_norm += similarity;
		}

		if (pos.y < map_size.y - step - 1) // Down
		{
			
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x*map_size.y*pos.z + map_size.x*(pos.y + step + 1) + pos.x];
			vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);
			float disp_check = sp_check.s7;

			diff = sqrt(pow(color_check.x - color.x, 2) + pow(color_check.y - color.y, 2) + pow(color_check.z - color.z, 2));
			similarity = exp(-diff*diff*gamma_i);
			diff = disp - disp_check;

			smoothness  += similarity * exp(-diff*diff*alpha);
			weight_norm += similarity;
			
		}
	}


	if (weight_norm > 0)
		return smoothness / weight_norm;
	else 
		return 0.000001;
}




void clDepthRefinement::init_state_per_element(float *current_state_host, int x, int y, int z, float gamma, float alpha, float fuse, int no_kernel_steps, int kernel_step_size)
{
	int idx_start = (6*map_size.x*map_size.y*z) + (6*map_size.x * y) + 6*x;
	int idx = (map_size.x*map_size.y*z) + (map_size.x*y) + x;
	
	vec8f current_spixl = spixl_map[idx];
	vec2f fl = flatness_map[idx];
	vec3i pos; pos.x = x; pos.y = y; pos.z = z;

	float d  = current_spixl.s7;
	float sm = init_smoothness(spixl_map, current_spixl, fl, map_size, pos, gamma, alpha, no_kernel_steps, kernel_step_size);
	float cs = init_consistency(view_subset, this->subset_num, pos, array_width, bl_ratio, fuse, alpha, gamma, fl);
	

	current_state_host[idx_start + 0] = d;
	current_state_host[idx_start + 1] = sm;
	current_state_host[idx_start + 2] = cs;
	current_state_host[idx_start + 3] = 0.0;
	current_state_host[idx_start + 4] = 0.0;
	current_state_host[idx_start + 5] = 1.0;

}




void clDepthRefinement::init_spixl_state_host(float *current_state_host, int start_view, int end_view, float gamma, float alpha, float fuse, int no_kernel_steps, float kernel_step_size)
{
	for (int z = start_view ; z < end_view ; z++)
		for (int y = 0 ; y < map_size.y ; y++)
			for (int x = 0 ; x < map_size.x ; x++)
				init_state_per_element(current_state_host, x, y, z, gamma, alpha, fuse, no_kernel_steps, kernel_step_size);
}



void clDepthRefinement::propagate(int iter, int norm, float gamma_, float alpha_, int kernel_steps_, int kernel_size_, float fuse_)
{
	printText("Propagate Info in Superpixel Map Iteration No: " + to_string(iter));

	// Load Context and Device //
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	
	// Set the Parameters //
	float gamma = 1.0 / gamma_;
	float alpha = 1.0 / alpha_;
	int no_kernel_steps = kernel_steps_;
	float kernel_step_size = std::max(1, kernel_size_ / no_kernel_steps * spixl_size);
	float fuse = 0.5 * fuse_;
		
	// Define Device Kernel //
	cl_int err;

	cl::Kernel kernel(program, "propagate");
	if (iter % 2 == 0)
	{
		err = kernel.setArg(0, *current_state_dev2); // output
		err = kernel.setArg(1, *current_state_dev);  // input
	}
	else
	{
		err = kernel.setArg(0, *current_state_dev); // output
		err = kernel.setArg(1, *current_state_dev2); // input
	}

	err = kernel.setArg(2, *spixl_map_dev);
	err = kernel.setArg(3, *idx_img_dev);
	err = kernel.setArg(4, *spixl_rep_dev);
	err = kernel.setArg(5, *flatness_map_dev);
	err = kernel.setArg(6, *view_subset_dev);
	err = kernel.setArg(7, *subset_num_dev);
	err = kernel.setArg(8, iter);
	err = kernel.setArg(9, map_size);
	err = kernel.setArg(10, img_size);
	err = kernel.setArg(11, alpha);
	err = kernel.setArg(12, gamma);
	err = kernel.setArg(13, bl_ratio);
	err = kernel.setArg(14, fuse);
	err = kernel.setArg(15, no_kernel_steps / (iter + 1) );
	err = kernel.setArg(16, kernel_step_size / (iter + 1) );
	err = kernel.setArg(17, array_width);
	err = kernel.setArg(18, norm);
	if (err != CL_SUCCESS) printError(err, "Set Arguman Error in Propagate Function");

	// configure Kernel //
	cl_int3 grid_size;
	grid_size.x = (int)(ceil((float)map_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)map_size.y / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.z = view_count;

	// Execute the Kernel //
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y, grid_size.z), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE, 1));
	queue.finish();

	if (err != CL_SUCCESS)	printError(err, "Execution Error in Propagate Function");


	// Test //

	//-------------------------------------------------------------------------------------------
	/**/

	if (iter % 2 == 0)
		err = queue.enqueueReadBuffer(*current_state_dev2, true, 0, 6 * map_size.x * map_size.y * view_count * sizeof(float), this->current_state2);
	else
		err = queue.enqueueReadBuffer(*current_state_dev, true, 0, 6 * map_size.x * map_size.y * view_count * sizeof(float), this->current_state);
	queue.finish();
	if (err != CL_SUCCESS) printError(err, "Read Buffer Error in Propagate Function");
	


	//--------------------------------------------------------------------------------------------
	/**/
	if (iter == 4)
	{
		//img_translate_state(current_state2, 0, view_count, 1, 10, 100, "5- smoothness/sm_dev");
		//img_translate_state(current_state2, 0, view_count, 2, 10, 100, "6- consistency/cs_dev");
		img_translate_state(current_state2, 0, view_count, 0, 30, 60, "7- propagate/change6_alter1");	
	}
	/**/	

	/**/



	//-----------------------------------------------------------------------------------------
	/**
	////////////// Propagate on Host //////////////
	printText("Propagate on Host");

	// Copy the Input: Initialize the Host Code
	if (iter == 0)
		memcpy(current_state_host, current_state, 6 * map_size.x * map_size.y * view_count * sizeof(float));

	if (iter % 2 == 0)
		propagate_host(current_state_host, current_state_host2, iter, norm, gamma, alpha, no_kernel_steps / (iter + 1), kernel_step_size / (iter + 1), fuse, 0, view_count);
	else 
		propagate_host(current_state_host2, current_state_host, iter, norm, gamma, alpha, no_kernel_steps / (iter + 1), kernel_step_size / (iter + 1), fuse, 0, view_count);
	//if (iter == 4)	img_translate_state(current_state_host2, 0, view_count, 0, 10, 100, "prop_host");
	/**/

	//----------------------------------------------------------------------------------------
	////////////////// Compare Output of Host and Device /////////////////
	/**
	

	if (iter == 0)	compare_host_to_device_state(current_state_host2, current_state2, 0, view_count, 1);
	if (iter == 4)	img_translate_state(current_state_host2, 0, view_count, 0, 10, 100, "prop_host2");
	//if (iter == 4)	img_translate_state(current_state_host2, 0, view_count, 0, 10, 100, "prop_host");
	/**/

		
	// Print
	/**
	if (iter == 4)
	{
		printText("Print Some Output from Both Host and Device: ");
		std::cout << std::endl;

		int view = 0;
		for (int i = 0 ; i < 10 ; i++)
		{
			for (int j = 0 ; j < 10 ; j++)
			{
				int idx = 6 * map_size.x* map_size.y * view + 6 * map_size.x * j + 6 * i;
				float val;
				if (iter % 2 == 0)
					val = current_state_host[idx + 2];
				else
					val = current_state_host2[idx + 2];
				std::cout << val << ", ";

			}
			cout << std::endl;
		}

		printText("Print Some Output from Device");

		for (int i = 0 ; i < 10 ; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				int idx = (6 * map_size.x * map_size.y * view) + (6 * map_size.x * j) + (6 * i);
				float val;
				if (iter % 2 == 0)
					val = current_state[idx + 2];
				else
					val = current_state2[idx + 2];
				std::cout << val << ", ";
			}
			cout << std::endl;
		}
	}
	

/**/		
	
/**/
}



float clDepthRefinement::compute_smoothness(float *current_state_host, vec3f color, vec2f center, float d, vec3f n, int x, int y, int z, float alpha, float gamma, int no_kernel_steps, float kernel_step_size, vec2f fl)
{
	float smoothness  = 0.0;
	float weight_norm = 0.0;


	// Check the Immidate Neighbors
	for (int i = -1 ; i <= 1 ; i++) for (int j = -1 ; j <= 1 ; j++)
	{
		vec3i pos_check = makeVec3i(x + i, y + j, z);

		if (pos_check.x >= 0 && pos_check.y >= 0 && pos_check.x < map_size.x && pos_check.y < map_size.y && (i != 0 || j != 0))
		{
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x * map_size.y * pos_check.z + map_size.x * pos_check.y + pos_check.x];
			vec3f color_check  = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);
			vec2f center_check = makeVec2f(sp_check.s1, sp_check.s2);

			diff = sqrt(pow(color_check.x - color.x, 2) + pow(color_check.y - color.y, 2) + pow(color_check.z - color.z, 2));
			similarity = exp(-diff * diff * gamma);


			float d_extp = (n.x * (center.x - sp_check.s1) + n.y * (center.y - sp_check.s2) + n.z * d) / n.z;
			
			diff = d_extp - current_state[(6 * map_size.x * map_size.y * pos_check.z) + (6 * map_size.x * pos_check.y) + 6 * pos_check.x];
			smoothness += similarity * exp(-diff * diff * alpha);
			weight_norm += similarity;
		}
	}


	int step_size = std::max(1, (int)(fl.x * kernel_step_size + 0.5));


	// Propagation Kernel 
	/**
	for (int i = 1; i <= no_kernel_steps; i++)
	{
		float gamma_i = gamma * (1 + i);
		int step = i * step_size;

		
		if (x > step) // Left
		{
			vec2i pos_check = makeVec2i(x - (step + 1), y);
			float diff, similarity;
		
			vec8f sp_check = spixl_map[map_size.x * map_size.y * z + map_size.x * pos_check.y + pos_check.x];
			vec2f center_check = makeVec2f(sp_check.s1, sp_check.s2);
			vec3f color_check  = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);

			diff = sqrt(pow(color_check.x - color.x, 2) + pow(color_check.y - color.y, 2) + pow(color_check.z - color.z, 2));
			similarity = exp(-diff*diff*gamma_i);

			float d_extp = (n.x * (center.x - sp_check.s1) + n.y * (center.y - sp_check.s2) + n.z * d) / n.z;


			diff = d_extp - current_state[(6 * map_size.x * map_size.y * z) + (6 * map_size.x * pos_check.y) + 6 * pos_check.x];

			smoothness += similarity * exp(-diff * diff * alpha);
			weight_norm += similarity;
		}

		
		if (x < map_size.x - step - 1) // Right
		{
			vec2i pos_check = makeVec2i(x + (step + 1), y);
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x * map_size.y * z + map_size.x * pos_check.y + pos_check.x];
			vec2f center_check = makeVec2f(sp_check.s1, sp_check.s2);
			vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);

			diff = sqrt(pow(color_check.x - color.x, 2) + pow(color_check.y - color.y, 2) + pow(color_check.z - color.z, 2));
			similarity = exp(-diff*diff*gamma_i);

			float d_extp = (n.x * (center.x - sp_check.s1) + n.y * (center.y - sp_check.s2) + n.z * d) / n.z;

			diff = d_extp - current_state[(6 * map_size.x * map_size.y * z) + (6 * map_size.x * pos_check.y) + 6 * pos_check.x];

			smoothness += similarity * exp(-diff * diff * alpha);
			weight_norm += similarity;
		}

		
		if (y > step) // UP
		{
			vec2i pos_check = makeVec2i(x, y - (step + 1));
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x * map_size.y * z + map_size.x * pos_check.y + pos_check.x];
			vec2f center_check = makeVec2f(sp_check.s1, sp_check.s2);
			vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);

			diff = sqrt(pow(color_check.x - color.x, 2) + pow(color_check.y - color.y, 2) + pow(color_check.z - color.z, 2));
			similarity = exp(-diff*diff*gamma_i);

			float d_extp = (n.x * (center.x - sp_check.s1) + n.y * (center.y - sp_check.s2) + n.z * d) / n.z;


			diff = d_extp - current_state[(6 * map_size.x * map_size.y * z) + (6 * map_size.x * pos_check.y) + 6 * pos_check.x];

			smoothness += similarity * exp(-diff * diff * alpha);
			weight_norm += similarity;
		}

		if (y < map_size.y - step - 1)
		{
			vec2i pos_check = makeVec2i(x, y + (step + 1));
			float diff, similarity;

			vec8f sp_check = spixl_map[map_size.x * map_size.y * z + map_size.x * pos_check.y + pos_check.x];
			vec2f center_check = makeVec2f(sp_check.s1, sp_check.s2);
			vec3f color_check  = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);

			diff = sqrt(pow(color_check.x - color.x, 2) + pow(color_check.y - color.y, 2) + pow(color_check.z - color.z, 2));
			similarity = exp(-diff*diff*gamma_i);

			float d_extp = (n.x * (center.x - sp_check.s1) + n.y * (center.y - sp_check.s2) + n.z * d) / n.z;


			diff = d_extp - current_state[(6 * map_size.x * map_size.y * z) + (6 * map_size.x * pos_check.y) + 6 * pos_check.x];

			smoothness += similarity * exp(-diff * diff * alpha);
			weight_norm += similarity;
		}
		
	}
	/**/

	if (weight_norm > 0)
		return smoothness / weight_norm;
	else
		return 0.000001;

}


float clDepthRefinement::compute_consistency(float *current_state_host, vec3f color, vec2f center, int x, int y, int z, float d, vec3f n, float alpha, float gamma, float bl_ratio, float fuse, vec2f fl)
{
	// Index
	int idx = map_size.x * map_size.y * z + map_size.x * y + x;

	// Set Parameters
	float consistency = 0.0;

	int view_counter = 0;
	vec2i camIdx;
	camIdx.x = z % array_width;
	camIdx.y = z / array_width;

	// Super pixels Samples
	vec8u rep = spixl_rep[z*map_size.x*map_size.y + y*map_size.x + x];

	int sp_samples[9];
	sp_samples[0] = (int)rep.s0;
	sp_samples[1] = (int)rep.s1;
	sp_samples[2] = (int)rep.s2;
	sp_samples[3] = (int)rep.s3;
	sp_samples[4] = 0;
	sp_samples[5] = (int)rep.s4;
	sp_samples[6] = (int)rep.s5;
	sp_samples[7] = (int)rep.s6;
	sp_samples[8] = (int)rep.s7;


	for (int k = 0 ; k < subset_num[z] ; k++)
	{
		int view = view_subset[view_count * z + k];

		vec2i viewIdx;
		viewIdx.x = view % array_width;
		viewIdx.y = view / array_width;

		// Consistency Variables
		float visib_weight_sum = 0.0f;
		float occl_weight_sum = 0.0;
		float num = 0.0;
		float visibility = 0.0;
		float visible = 0.0;

		for (int i = -1 ; i <= 1 ; i++) for (int j = -1 ; j <= 1 ; j++)
		{
			// Take One Sample Point at a Time
			vec2i xy;
			xy.x = (int)center.x + sp_samples[(i + 1) * 3 + j + 1] * i;
			xy.y = (int)center.y + sp_samples[(i + 1) * 3 + j + 1] * j;

			// Interpolate the Sample point in the Plane Equ
			float d_intrp = (n.x * (center.x - xy.x) + n.y * (center.y - xy.y) + n.z * d) / n.z;

			// Project the sample point in the Current Subset View 
			vec2i xy_proj;
			xy_proj.x = xy.x - round( d_intrp * (viewIdx.x - camIdx.x));
			xy_proj.y = xy.y - round(bl_ratio * d * (viewIdx.y - camIdx.y));

			if (xy_proj.x >= 0 && xy_proj.y >= 0 && xy_proj.x < img_size.x  && xy_proj.y < img_size.y)
			{
				uint idx_proj = idx_img[img_size.x*img_size.y*view + img_size.x*xy_proj.y + xy_proj.x];
				vec2i coord_proj;
				coord_proj.x = idx_proj % map_size.x;
				coord_proj.y = idx_proj / map_size.x;

				// Load Proj Superpixel Info
				vec8f sp_proj = spixl_map[map_size.x*map_size.y*view + map_size.x*coord_proj.y + coord_proj.x];
				vec2f center_proj = makeVec2f(sp_proj.s1, sp_proj.s2);
				vec3f color_proj = makeVec3f(sp_proj.s3, sp_proj.s4, sp_proj.s5);

				// Load Proj Superpixl State
				int state_idx_proj = (6 * map_size.x * map_size.y * view) + (6 * map_size.x * coord_proj.y) + (6 * coord_proj.x);
				float d_proj = current_state[state_idx_proj];
				vec3f n_proj = makeVec3f(current_state[state_idx_proj + 3], current_state[state_idx_proj + 4], current_state[state_idx_proj + 5]);
				
				// Interpolate Proj Sample Point in Plane Equ
				float d_intrp_proj = (n_proj.x*(center_proj.x - xy_proj.x) + n_proj.y*(center_proj.y - xy_proj.y) + n_proj.z*d_proj) / n_proj.z;

				float diff = d_intrp_proj - d;
				float when_visible = 0.0;
				if (abs(diff) < fuse)  when_visible = 1.0;
				visible += when_visible*exp(-diff*diff*alpha);
				visib_weight_sum += when_visible;
				occl_weight_sum += (1 - when_visible);

				diff = sqrt(pow(color_proj.x - color.x, 2) + pow(color_proj.y - color.y, 2) + pow(color_proj.z - color.z, 2));
				visibility += exp(-diff * diff * gamma);

				num += 1;
			}
		}

		if (num > 0)
		{
			view_counter++;
			if (visib_weight_sum > 0)
				consistency += (visib_weight_sum / num) * (visibility / visib_weight_sum) * (visible / visib_weight_sum);

			if (occl_weight_sum > 0)
				consistency += 0.5*fl.y;
		}
	}

	float margin = 0.01;

	if (view_counter > 0)
		return max(margin, consistency / view_counter);
	else
		return margin;
}


vec8f clDepthRefinement::update_current_thread(float *current_state_in, int idx_check, int idx_state_check, vec2f center, vec3f color, int iter, vec3f n0, float sm0, float cs0, float d0, float alpha, float gamma, float fuse, float bl_ratio, vec2f fl, int x, int y, int z, int no_kernel_steps, float kernel_step_size)
{

	vec3f n1 = makeVec3f(current_state_in[idx_state_check + 3], current_state_in[idx_state_check + 4], current_state_in[idx_state_check + 5]);
	float d1 = current_state_in[idx_state_check];

	// Load Neighbor Sp Info
	vec8f sp_check = spixl_map[idx_check];
	vec3f color_check = makeVec3f(sp_check.s3, sp_check.s4, sp_check.s5);
	vec2f center_check = makeVec2f(sp_check.s1, sp_check.s2);

	// Interpolate	
	float d_intrp = (n1.x * (center_check.x - center.x) + n1.y * (center_check.y - center.y) + n1.z * d1) / n1.z;

	// Compute New Smoothnees and Consistency
	float sm1 =  compute_smoothness(current_state_in, color, center, d_intrp, n1, x, y, z, alpha, gamma, no_kernel_steps, kernel_step_size, fl);
	float cs1 =  compute_consistency(current_state_in, color, center, x, y, z, d_intrp, n1, alpha, gamma, bl_ratio, fuse, fl);
	
	// Update: 
	float diff = euDistance3D(color, color_check);
	float similarity = exp(-diff * diff * gamma);

	float sm_update = sm0, cs_update = cs0, d_update = d0;
	vec3f n_update = n0;
	/**/
	if ((iter < 4 && sm1 > sm0) || cs1 * sm1 > sm0 * cs0)
	{
		d_update = d_intrp;
		sm_update = sm1;
		cs_update = cs1;
		n_update = n1;
	}
	/**/

	vec8f update_state = makeVec8f(d_update, sm_update, cs_update, n_update.x, n_update.y, n_update.z, 0, 0);
	return update_state;
}


vec8f clDepthRefinement::refine_current_thread(float *current_state_in, vec2i nbr1, vec2i nbr2, int x, int y, int z, vec2f center, vec3f color, float d0, float sm0, float cs0, vec3f n0, vec2f fl, float alpha, float gamma, float bl_ratio, float fuse, int no_kernel_steps, float kernel_step_size, int iter)
{
	vec3f center1;
	center1.x = spixl_map[map_size.x * map_size.y * z + map_size.x * nbr1.y + nbr1.x].s1;
	center1.y = spixl_map[map_size.x * map_size.y * z + map_size.x * nbr2.y + nbr2.x].s2;
	center1.z = current_state_in[(6 * map_size.x * map_size.y * z) + (6 * map_size.x * nbr1.y) + (6 * nbr1.x)];

	vec3f center2;
	center2.x = spixl_map[map_size.x * map_size.y * z + map_size.x * nbr2.y + nbr2.x].s1;
	center2.y = spixl_map[map_size.x * map_size.y * z + map_size.x * nbr2.y + nbr2.x].s2;
	center2.z = current_state_in[(6 * map_size.x * map_size.y * z) + (6 * map_size.x * nbr2.y) + (6 * nbr2.x)];
	
	vec3f v1 = makeVec3f(center1.x - center.x, center1.y - center.y, center1.z - d0);
	vec3f v2 = makeVec3f(center2.x - center.x, center2.y - center.y, center2.z - d0);

	vec3f n1 = normalizeVec3f(crossVec3f(v1, v2) );

	// Compute New Smoothnees and Consistency
	float sm1 = compute_smoothness(current_state_in, color, center, d0, n1, x, y, z, alpha, gamma, no_kernel_steps, kernel_step_size, fl);
	float cs1 = compute_consistency(current_state_in, color, center, x, y, z, d0, n1, alpha, gamma, bl_ratio, fuse, fl);

	// Update: 

	float sm_update = sm0, cs_update = cs0, d_update = d0;
	vec3f n_update = n0;

	if ((iter < 4 && sm1 > sm0) || cs1 * sm1 > sm0 * cs0)
	{
		sm_update = sm1;
		cs_update = cs1;
		n_update = n1;
	}

	vec8f update_state = makeVec8f(d_update, sm_update, cs_update, n_update.x, n_update.y, n_update.z, 0, 0);
	return update_state;
}


void clDepthRefinement::propagate_current_thread(float *current_state_in, float *current_state_out, int iter, int norm, float gamma, float alpha, int no_kernel_steps, float kernel_step_size, float fuse, int x, int y, int z)
{
	int state_idx = (6 * map_size.x*map_size.y * z) + (6 * map_size.x * y) + 6 * x;

	int idx = map_size.x * map_size.y * z  +  map_size.x * y  +  x;

	// Load Sp Info
	vec8f sp = spixl_map[idx];
	vec2f center = makeVec2f(sp.s1, sp.s2); 	
	vec3f color  = makeVec3f(sp.s3, sp.s4, sp.s5);

	// Load Sp State
	float d0  = current_state_in[state_idx];
	float sm0 = current_state_in[state_idx + 1];
	float cs0 = current_state_in[state_idx + 2];

	// Normal Vector
	vec3f n0 = makeVec3f(current_state_in[state_idx + 3], current_state_in[state_idx + 4], current_state_in[state_idx + 5]);

	// Load Inputs
	vec2f fl = flatness_map[idx];
	vec8u sp_rep = spixl_rep[idx];

	// Propagate Info for Immidiate Neighbors
	//for (int i = -1 ; i <= 1 ; i++) for (int j = -1 ; j <= 1 ; j++)
	for (int i = -1 ; i <= 1 ; i++) for (int j = -1 ; j <= 1 ; j++)
	{
		vec3i p = makeVec3i(x + i, y + j, z);
		if (p.x >= 0 && p.y >= 0 && p.x < map_size.x && p.y < map_size.y && !(i == 0 && j == 0))
		{
			int state_idx_check = (6 * map_size.x * map_size.y * p.z) + (6 * map_size.x * p.y) + (6 * p.x);

			int idx_check = (map_size.x * map_size.y * p.z) + (map_size.x * p.y) + p.x;

			// Update Current Plane
			/**/
			vec8f updated_surface = update_current_thread(current_state_in, idx_check, state_idx_check, center, color, iter, n0, sm0, cs0, d0, alpha, gamma, fuse, bl_ratio, fl, x, y, z, no_kernel_steps, kernel_step_size);

			d0  = updated_surface.s0;
			sm0 = updated_surface.s1;
			cs0 = updated_surface.s2;
			n0 = makeVec3f(updated_surface.s3, updated_surface.s4, updated_surface.s5);
			/**/
		}
	}

	////////////////////
	//// Refinement ///
	//////////////////
	/**
	vec2i nbr[8];
	nbr[0] = makeVec2i(x - 1, y);
	nbr[1] = makeVec2i(x - 1, y - 1);
	nbr[2] = makeVec2i(x, y - 1);
	nbr[3] = makeVec2i(x + 1, y - 1);
	nbr[4] = makeVec2i(x + 1, y);
	nbr[5] = makeVec2i(x + 1, y + 1);
	nbr[6] = makeVec2i(x, y + 1);
	nbr[7] = makeVec2i(x - 1, y + 1);

	for (int i = 0 ; i < 8 ; i++)
	{
		int j = (i + 1) % 8;
		vec8f updated_plane;
		if (nbr[i].x > -1 && nbr[i].y > -1 && nbr[i].x < map_size.x && nbr[i].y < map_size.y && nbr[j].x > -1 && nbr[j].y > -1 && nbr[j].x < map_size.x && nbr[j].y < map_size.y)
		{ 
			updated_plane = refine_current_thread(current_state_in, nbr[i], nbr[j], x, y, z, center, color, d0, sm0, cs0, 
																				n0, fl, alpha, gamma, bl_ratio, fuse, no_kernel_steps, kernel_step_size, iter);
			d0  = updated_plane.s0;
			sm0 = updated_plane.s1;
			cs0 = updated_plane.s2;
			n0  = makeVec3f(updated_plane.s3, updated_plane.s4, updated_plane.s5);
		}
	}

	/**/
	current_state_out[(map_size.y * 6 * map_size.x * z) + (6 * map_size.x * y) + (6 * x + 0)] = d0;
	current_state_out[(map_size.y * 6 * map_size.x * z) + (6 * map_size.x * y) + (6 * x + 1)] = sm0;
	current_state_out[(map_size.y * 6 * map_size.x * z) + (6 * map_size.x * y) + (6 * x + 2)] = cs0;

	current_state_out[(map_size.y * 6 * map_size.x * z) + (6 * map_size.x * y) + (6 * x + 3)] = n0.x;
	current_state_out[(map_size.y * 6 * map_size.x * z) + (6 * map_size.x * y) + (6 * x + 4)] = n0.y;
	current_state_out[(map_size.y * 6 * map_size.x * z) + (6 * map_size.x * y) + (6 * x + 5)] = n0.z;
}



void clDepthRefinement::propagate_host(float *current_state_in, float *current_state_out, int iter, int norm, float gamma, float alpha, int no_kernel_steps, float kernel_step_size, float fuse, int start_view, int end_view)
{
	#pragma omp parallel for
	for (int z = start_view ; z < end_view ; z++)
		for (int y = 0 ; y < map_size.y ; y++)
			for (int x = 0 ; x < map_size.x ; x++)
					propagate_current_thread(current_state_in, current_state_out, iter, norm, gamma, alpha, no_kernel_steps, kernel_step_size, fuse, x, y, z);
}


void clDepthRefinement::fusion(float alpha_, float gamma_, float bl_ratio_, float fuse_)
{
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
	printText("Fusion Function");

	// Setting up the parameters //
	float gamma = 1 / gamma_;
	float alpha = 1 / alpha_;
	float bl_ratio = bl_ratio_;
	float fuse = 0.5 * fuse_;

	// ----------------------------------
	cl_int err; 

	disp_full_dev = new cl::Buffer(context, CL_MEM_READ_WRITE,
		img_size.x * img_size.y * view_count * sizeof(float), nullptr, &err);

	if (err != CL_SUCCESS){
		printText("ERROR: FUSION FUNCTION BUFFER ALLOCATION. ERROR NO: : " + to_string(err));
		return;
	}

	// ------------------------------------
	// Kernel Configuration
	cl_int3 grid_size;
	grid_size.x = (int)(ceil((float)img_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)img_size.y / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.z = view_count;

	// Kernel 1 Define:
	cl::Kernel kernel_1(program, "spixl_to_image");
	err = kernel_1.setArg(0, *spixl_map_dev);
	err = kernel_1.setArg(1, *idx_img_dev);
	err = kernel_1.setArg(2, *current_state_dev);
	err = kernel_1.setArg(3, *disp_full_dev);
	err = kernel_1.setArg(4, gamma);
	err = kernel_1.setArg(5, img_size);
	err = kernel_1.setArg(6, map_size);
	if (err != CL_SUCCESS) {
		printText("ERROR: FUSION FUNCTION SET ARGUMAN. ERROR NO: " + to_string(err));
		return;
	}
	

	// Exe Kernel_1
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y, grid_size.z), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE, 1));
	queue.finish();
	if (err != CL_SUCCESS) {
		printText("ERROR: FUSION FUNCTION FIRST KERNEL EXE ERROR. ERROR NO: " + to_string(err));
		return;
	}

	// ---------------------------------------------------------
	// Release Super-pixel Map Device Memory Objects
	/**
	delete idx_img_dev;
	delete spixl_map_dev;
	delete cvt_img_dev;
	delete current_state_dev;
	delete current_state_dev2;
	delete spixl_rep_dev;

	disp_proj_dev = new cl::Buffer(context, CL_MEM_READ_WRITE,
		img_size.x * img_size.y * view_count * sizeof(float), nullptr, &err);

	cl::Buffer disp_vol_dev(context, CL_MEM_READ_WRITE,
		img_size.x * img_size.y * view_count * sizeof(float), nullptr, &err);

	disp_img_dev = new cl::Buffer(context, CL_MEM_WRITE_ONLY,
		img_size.x * img_size.y * view_count * sizeof(float), nullptr, &err);

	if (err != CL_SUCCESS) {
		printText("ERROR: FUSION FUNCTION FIRST KERNEL EXE ERROR. ERROR NO: " + to_string(err));
		return;
	}


	// Take Each Image as Reference in Turn
	for (int i = 0 ; i < view_count ; i++)
	{
		// ------------------------------------------------------
		// Def Kernel_2
		
		cl::Kernel kernel_2(program, "project_to_reference_inv");
		err = kernel_2.setArg(0, *disp_full_dev);
		err = kernel_2.setArg(1, *disp_proj_dev);
		err = kernel_2.setArg(2, i);
		err = kernel_2.setArg(3, img_size);
		err = kernel_2.setArg(4, bl_ratio);
		err = kernel_2.setArg(5, array_width);
		err = kernel_2.setArg(6, view_count);
		if (err != CL_SUCCESS) {
			printText("ERROR: FUSION FUNCTION KERNEL_2 SET ARGUMAN. ERROR NO: " + to_string(err));
			return;
		}

		// Exe Kernel_2
		err = queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE));
		queue.finish();
		if (err != CL_SUCCESS) {
			printText("ERROR: FUSION FUNCTION SECOND KERNEL EXE ERROR. ERROR NO: " + to_string(err));
			return;
		}
		
		// -----------------------------------
		// Def Kernel_3
		
		cl::Kernel kernel_3(program, "remove_view_inconsistency");
		err = kernel_3.setArg(0, *disp_proj_dev);
		err = kernel_3.setArg(1, *disp_full_dev);
		err = kernel_3.setArg(2, *subset_num_dev);
		err = kernel_3.setArg(3, *view_subset_dev);
		err = kernel_3.setArg(4, i);
		err = kernel_3.setArg(5, bl_ratio);
		err = kernel_3.setArg(6, fuse);
		err = kernel_3.setArg(7, img_size);
		err = kernel_3.setArg(8, array_width);
		err = kernel_3.setArg(9, view_count);
		err = kernel_3.setArg(10, *disp_img_dev);
		if (err != CL_SUCCESS) {
			printText("ERROR: FUSION FUNCTION KERNEL_3 SET ARGUMAN. ERROR NO: " + to_string(err));
			return;
		}

		// Exe Kernel
		err = queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE));
		queue.finish();
		if (err != CL_SUCCESS) {
			printText("ERROR: FUSION FUNCTION SECOND KERNEL EXE ERROR. ERROR NO: " + to_string(err));
			return;
		}
	}

	/**/

	// -----------------------------------------------------
	// Update from Device to Host:
	/**/
	disp_img = new float[img_size.x * img_size.y * view_count]; // Allocate mem obj on host
	err = queue.enqueueReadBuffer(*disp_full_dev, false, 0, img_size.x * img_size.y * view_count * sizeof(float), disp_img);
	if (err != CL_SUCCESS) {
		printText("ERROR: FUSION FUNCTION READ FROM BUFFER. ERROR NO: " + to_string(err));
		return;
	}

	// Plot
	plot_full_image(0, view_count, 30, 60, "8- Fusion/fus4");

	//delete disp_img;	// Release the memory obj on host
	/**/
}


void clDepthRefinement::plot_full_image(int start_view, int end_view, int min_disp, int max_disp, std::string file_name)
{
	Mat test_cam(img_size.y, img_size.x, CV_32FC1);
	Mat test_cam_2(img_size.y, img_size.x, CV_8UC1);
	printText("Plot Fusion Function: Writing the Results into File");

	for (int k = start_view ; k < end_view ; k++)
	{
		for (int i = 0; i < img_size.y; i++)
		{
			for (int j = 0; j < img_size.x; j++)
			{
				int idx = img_size.x*img_size.y*k + img_size.x*i + j;
				float d = disp_img[idx];
				unsigned char d_scale = (unsigned char)floor(((d - min_disp) / (max_disp - min_disp)) * 255);
				test_cam_2.at<unsigned char>(i, j) = d_scale;
			}
		}

		std::string address = "../results/" + file_name + " " + std::to_string(k) + ".png";
		imwrite(address, test_cam_2);
	}
}


void clDepthRefinement::img_translate_state(float *current_state, int start_view, int end_view, int element_num, int min_disp, int max_disp, std::string file_name)
{

	Mat test_cam(img_size.y, img_size.x, CV_32FC1);
	Mat test_cam_2(img_size.y, img_size.x, CV_8UC1);
	//namedWindow("My Window", WINDOW_AUTOSIZE);

	for (int k = start_view ; k < end_view ; k++)
	{
		for (int i = 0 ; i < img_size.y ; i++)
			for (int j = 0 ; j < img_size.x ; j++)
			{
				int sp_idx = idx_img[img_size.x*img_size.y*k + img_size.x*i + j];
				int sp_x = sp_idx % map_size.x;
				int sp_y = sp_idx / map_size.x;

				int state_idx = (map_size.x * 6 * map_size.y*k) + (map_size.x * 6 * sp_y) + (6 * sp_x);
				
				

				switch (element_num)
				{
					case 0:
					{
						float d = current_state[state_idx + 0];
						unsigned char d_scale = (unsigned char)floor(((d - min_disp) / (max_disp - min_disp)) * 255);
						test_cam.at<float>(i, j) = (d - 10) / 90;
						test_cam_2.at<unsigned char>(i, j) = d_scale;
						break;
					}

					case 1:
					{
						float sm = current_state[state_idx + 1];
						unsigned char sm_scale = (unsigned char)floor(sm * 255);
						test_cam.at<float>(i, j) = sm;
						test_cam_2.at<unsigned char>(i, j) = sm_scale;
						break;
					}

					case 2:
					{
						float cs = current_state[state_idx + 2];
						unsigned char cs_scale = (unsigned char)floor(cs * 255);
						test_cam.at<float>(i, j) = cs;
						test_cam_2.at<unsigned char>(i, j) = cs_scale;
						break;
					}
				}
			}
	

		std::string address = "../results/" + file_name + " " + std::to_string(k) + ".png";
		imwrite(address, test_cam_2);
		//imshow("My Window", test_cam);
		//waitKey(0);
	}

}
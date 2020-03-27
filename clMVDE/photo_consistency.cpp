#include "stdafx.h"
#include "photo_consistency.h"
#include "test.h"


clPhotoConsistency::clPhotoConsistency(cl::Program program_, int view_count_, int spixl_size_, int num_disp_levels_, vec2i img_size_, vec2i map_size_)
{
	this->program = program_;
	this->view_count = view_count_;
	this->spixel_size = spixl_size_;
	this->num_disp_levels = num_disp_levels_;
	this->img_size = img_size_;
	this->map_size = map_size_;

}





void clPhotoConsistency::do_initial_depth_estimation(
	vec8f *spixel_map,
	vec8u *spixel_rep,
	vec3f *in_img,
	cl_uint   *idx_img,
	int array_width,
	float bl_ratio,
	vector<vector<int> > &view_subset,
	vector<float> &disp_levels
)
{
	// Load Device & Context
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	no_disp = disp_levels.size();

	// Reading Neighboring Cameras
	int *subset_num_per_cam = new int[view_count];
	int *view_subset_mat = new int[view_count * view_count];

	for (int i = 0 ; i < view_count ; i++)
	{
		subset_num_per_cam[i] = view_subset[i].size();
		for (int j = 0 ; j < view_subset[i].size() ; j++)
			view_subset_mat[i * view_count + j] = view_subset[i][j];
	}


	

	cl_int err;
	cl::Buffer disp_levels_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		num_disp_levels * sizeof(float), disp_levels.data(), &err);

	cl::Buffer view_subset_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		view_count * view_count * sizeof(int), view_subset_mat, &err);

	cl::Buffer subset_num_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		view_count * sizeof(int), subset_num_per_cam, &err);

	//delete view_subset_mat; delete subset_num_per_cam;

	cl::Buffer cvt_img_dev(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		img_size.x * img_size.y * view_count * sizeof(vec3f), in_img, &err);

	cl::Buffer spixel_map_dev(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		map_size.x * map_size.y * view_count * sizeof(vec8f), spixel_map, &err);

	cl::Buffer spixel_rep_dev(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
		map_size.x * map_size.y * view_count * sizeof(vec8u), nullptr, &err);	////bug bug bug 

	cl::Buffer idx_img_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		img_size.x * img_size.y * view_count * sizeof(cl_uint), idx_img, &err);
	if (err != CL_SUCCESS)	cout << "PhotoConsistency Memory Allocate Error. Error Number: " << err << endl;
	
	

	// for test //
	//vec8f *spixl_map_host = new vec8f[map_size.x * map_size.y * view_count];
	//memcpy(spixl_map_host, spixel_map, map_size.x * map_size.y * view_count * sizeof(vec8f));
	//////// 
	

	// Config kernel
	/**/
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float) map_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float) map_size.y / LOCAL_SIZE)) * LOCAL_SIZE;

	// Def Queue
	cl::CommandQueue queue(context, devices[0]);

	// Def kernel 1
	cl::Kernel kernel_1(program, "find_super_pixel_boundary");
	err = kernel_1.setArg(0, spixel_map_dev);
	err = kernel_1.setArg(1, idx_img_dev);
	err = kernel_1.setArg(2, spixel_rep_dev);
	err = kernel_1.setArg(3, map_size);
	err = kernel_1.setArg(4, img_size);
	err = kernel_1.setArg(5, spixel_size);
	if (err != CL_SUCCESS) std::cout << " PhotoConsistency Arguman Error. Error Number: " << err << std::endl;

	// Exe kernel 1
	err = queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y, view_count), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE, 1));
	queue.finish();
	if (err != CL_SUCCESS)	std::cout << " PhototConsistency Kernel_1 Call Error. Error Number: " << err << std::endl;

	// Trans Device to Host
	err = queue.enqueueReadBuffer(spixel_rep_dev, CL_TRUE, 0, map_size.x * map_size.y * view_count * sizeof(vec8u), spixel_rep); queue.finish();
	if (err != CL_SUCCESS)	cout << " PhotoConsistency kernel_1 Read Buffer Error. Error Number: " << err << endl;

	// Def Kernel 2
	cl::Kernel kernel_2(program, "initial_depth_estimation_v2");
	err = kernel_2.setArg(0, cvt_img_dev);
	err = kernel_2.setArg(1, spixel_map_dev);
	err = kernel_2.setArg(2, spixel_rep_dev);
	err = kernel_2.setArg(3, idx_img_dev);
	err = kernel_2.setArg(4, disp_levels_dev);
	err = kernel_2.setArg(5, view_subset_dev);
	err = kernel_2.setArg(6, subset_num_dev);
	err = kernel_2.setArg(7, array_width);
	err = kernel_2.setArg(8, map_size);
	err = kernel_2.setArg(9, img_size);
	err = kernel_2.setArg(10, bl_ratio);
	err = kernel_2.setArg(11, spixel_size);
	err = kernel_2.setArg(12, no_disp);
	err = kernel_2.setArg(13, view_count);

	if (err != CL_SUCCESS)	std::cout << "Error in Photoconsistency. Error no: " << err << ". Set Arguman Error." << std::endl;

	// Exe kernel 2
	for (int z = 0 ; z < view_count ; z++)
	{
		std::cout << "Initialize " << z << "th View" << std::endl;
		err = kernel_2.setArg(14, z);
		err = queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y, 1), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE, 1));
		queue.finish();
		if (err != CL_SUCCESS)	std::cout << "Error in Photoconsistency. Error no: " << err << ". Kernel 2 Call Error." << std::endl;
	}
	

	// Transfer Data from Device to Host 
	err = queue.enqueueReadBuffer(spixel_map_dev, CL_TRUE, 0, map_size.x * map_size.y * view_count * sizeof(cl_float8), spixel_map); queue.finish();
	if (err != CL_SUCCESS)	cout << " PhotoConsistency kernel_2 Read Buffer Error. Error Number: " << err << endl;
	std::cout << "end of photoconsistency " << std::endl;
	img_translate(idx_img, spixel_map, 30, 60, 0, view_count);

	/**/
	
	//img_translate(idx_img, spixel_map, 10, 100, 0, 15);

	/** // test 1
	vec8u *spixl_rep_host = new vec8u[map_size.x * map_size.y * view_count];
	//vec2i *steps_host = new vec2i[map_size.x * map_size.y * view_count];
	vec8f *spixl_map_host = new vec8f[map_size.x * map_size.y * view_count];
	memcpy(spixl_map_host, spixel_map, map_size.x * map_size.y * view_count * sizeof(vec8f));
	
	this->do_initial_depth_estimation_host(spixl_map_host, spixl_rep_host, in_img, idx_img, array_width, bl_ratio, 
																								disp_levels, view_subset_mat, subset_num_per_cam, 0, 15);
	//compare_host_to_device(spixel_map, spixl_map_host, 0, 15);
	img_translate(idx_img, spixl_map_host, 10, 100, 0, 15);

	/**/
	/**
	int zero_count = 0;
	int count = 0;
	for (int i = 0 ; i < map_size.y ; i++)
	{
		for (int j = 0 ; j < map_size.x ; j++)
		{
			if (spixl_map_host[map_size.x*map_size.y*1 + map_size.x*i + j].s7 == 0)
				zero_count++;
			if (spixl_map_host[map_size.x*map_size.y * 1 + map_size.x*i + j].s7 == 10)
				count++;
		}
			//std::cout << spixl_map_host[map_size.x*map_size.y*1 + map_size.x*i + j].s7 << ", ";
		//std::cout << std::endl;
	}
	std::cout << "zero count = " << zero_count << std::endl;
	std::cout << "count = " << count << std::endl;
	//compare_host_to_device(spixel_map, spixl_map_host);

	/**/

	
	/** //test 2
	
	int start = 50, end = 70;
	cout << " Host: " << std::endl;
	for (int y = start ; y < end ; y++)
	{
		for (int x = start ; x < end ; x++)
			std::cout << spixel_map[1*map_size.x*map_size.y + y*map_size.x + x].s7<<", ";
		std::cout << std::endl;
	}

	
	cout << " Device: " << std::endl;
	for (int y = start ; y < end ; y++)
	{
		for (int x = start ; x < end ; x++)
			std::cout << (int)steps[0 * map_size.x * map_size.y + y * map_size.x + x].s0<<", ";
		cout << std::endl;
	}
	/**/

}



void clPhotoConsistency::compare_host_to_device(vec8f *a_d, vec8f *a_h, int start_view, int end_view)
{
	int miss = 0;
	int zero_count = 0, non_zero_count = 0;

	for (int z = start_view ; z < end_view ; z++)
		for (int y = 0 ; y < map_size.y ; y++)
			for (int x = 0 ; x < map_size.x ; x++)
			{
				int idx = z * map_size.x * map_size.y + y * map_size.x + x;
				if (a_d[idx].s7 != a_h[idx].s7)
				{
					miss++;
					if (miss < 10)
						std::cout << "x = "<<x <<" y = "<< y<<", a_d[" << idx << "] = " << a_d[idx].s7 << ", a_h[" << idx << "] = " << a_h[idx].s7 << std::endl;
				}	
				if (a_d[idx].s7 == 0.0)
					zero_count++;
				else
					non_zero_count++;
			}

	std::cout << "miss = " << miss << std::endl;
	std::cout << "zero_count percentage = " << ((float)zero_count / (float)(non_zero_count + zero_count))*100 << std::endl;
}



void clPhotoConsistency::init_depth_map(vec8f *spixl_map, vec8u *spixl_rep, vec3f *cvt_img, cl_uint *idx_img, int *view_subset_num,
	int *view_subset_vec, float *disp_levels, int x, int y, int z)
{
	int idx = z*map_size.x*map_size.y + y*map_size.x + x;

	vec8f spixl = spixl_map[idx];
	vec2i centeri; centeri.x = (int)(spixl.s1); centeri.y = (int)(spixl.s2);
	vec2f center; center.x = spixl.s1; center.y = spixl.s2;
	/**
	if (center.x < spixel_size)
		center.x += (spixel_size - center.x);

	if (center.x + spixel_size > img_size.x)
		center.x -= spixel_size;

	if (center.y < spixel_size)
		center.y += (spixel_size - center.y);

	if (center.y + spixel_size > img_size.y)
		center.y -= spixel_size;

	/**/

	vec8u dir;
	dir.s0 = 0; dir.s1 = 0; dir.s2 = 0; dir.s3 = 0; dir.s4 = 0; dir.s5 = 0; dir.s6 = 0; dir.s7 = 0;
	int base = z*img_size.x*img_size.y;

	/**/

	for (int i = 1 ; i < spixel_size ; i++)
	{
		// Check NW
		if (centeri.x - i >= 0 && centeri.y - i >= 0)
			if (idx == idx_img[base + (centeri.y - i)*img_size.x + centeri.x - i])
				dir.s0 = (uchar)(i - 1);


		// Check W
		if (centeri.x - i >= 0)
			if (idx == idx_img[base + centeri.y*img_size.x + centeri.x - i])
				dir.s1 = (uchar)(i - 1);

		// Check SW
		if (centeri.x - i >= 0 && centeri.y + i < img_size.y)
			if (idx == idx_img[base + (centeri.y + i)*img_size.x + centeri.x - i])
				dir.s2 = (uchar)(i - 1);

		// Check N
		if (centeri.y - i >= 0)
			if (idx == idx_img[base + (centeri.y - i)*img_size.x + centeri.x])
				dir.s3 = (uchar)(i - 1);
		// Check S
		if (centeri.y + i < img_size.y)
			if (idx == idx_img[base + (centeri.y + i)*img_size.x + centeri.x])
				dir.s4 = (uchar)(i - 1);

		// Check NE
		if (centeri.x + i < img_size.x && centeri.y - i >= 0)
			if (idx == idx_img[base + (centeri.y - i)*img_size.x + centeri.x + i])
				dir.s5 = (uchar)(i - 1);

		// Check E
		if (centeri.x + i < img_size.x)
			if (idx == idx_img[base + (centeri.y)*img_size.x + centeri.x + i])
				dir.s6 = (uchar)(i - 1);

		// Check SE
		if (centeri.x + i < img_size.x && centeri.y + i < img_size.y )
			if (idx == idx_img[base + (centeri.y + i)*img_size.x + centeri.x + i])
				dir.s7 = (uchar)(i - 1);
	}

	spixl_rep[idx] = dir;
	/**/
	/**/
	int bb_l = max((int)dir.s0, max((int)(dir.s1), (int)(dir.s2) ) );
	int bb_r = max((int)dir.s5, max((int)(dir.s6), (int)(dir.s7) ) );
	int bb_t = max((int)dir.s0, max((int)(dir.s3), (int)(dir.s5) ) );
	int bb_b = max((int)dir.s2, max((int)(dir.s4), (int)(dir.s7) ) );

	int step_x = max(1.0, 0.25*(bb_l + bb_r) );
	int step_y = max(1.0, 0.25*(bb_t + bb_b) );
	/**/


	int array_width = 5;
	float bl_ratio = 0.625;
	//vec2i step; step.x = 1; step.y = 1;
	vec2f step; step.x = (float)step_x; step.y = (float)step_y;

	//float d = 1.0;
	float cost_est = 1000000.0, disp_est = 0.0;
	float T = 30.0;
	
	for (int dl = 0 ; dl < no_disp ; dl++)
	{
		float d = disp_levels[dl];

		float min_val = 1000000.0;
		for (int n = 0 ; n < view_subset_num[z] ; n++)
		{
			float val = 0.0;
			int view = view_subset_vec[z*view_count + n];

			vec2i viewIdx, camIdx;
			viewIdx.x = view % array_width; 
			viewIdx.y = view / array_width;
			camIdx.x = z % array_width; camIdx.y = z / array_width;

			for (int i = 0 ; i <= 4 ; i++) for (int j = 0 ; j <= 4 ; j++)
			{
				vec2i xy_ref;
				xy_ref.x = center.x - 2 * step.x + i*step.x;
				xy_ref.y = center.y - 2 * step.y + j*step.y;

				vec2i xy_proj;
				xy_proj.x = (int)((float)xy_ref.x - d*(float)(viewIdx.x - camIdx.x));
				xy_proj.y = (int)((float)xy_ref.y - bl_ratio*d*(float)(viewIdx.y - camIdx.y) );

				if (xy_ref.x >= 0 && xy_ref.y >= 0 && xy_proj.x >= 0 && xy_proj.y >= 0 && xy_ref.x < img_size.x && xy_ref.y < img_size.y && xy_proj.x < img_size.x  && xy_proj.y < img_size.y)
				{
					vec3f color_ref  = cvt_img[img_size.x*img_size.y*z + img_size.x*xy_ref.y + xy_ref.x];
					vec3f color_proj = cvt_img[img_size.x*img_size.y*view + img_size.x*xy_proj.y + xy_proj.x];

					val += abs(color_ref.x - color_proj.x) + abs(color_ref.y - color_proj.y) + abs(color_ref.z - color_proj.z);
				}
				else
					val += T;
			}
			if (val < min_val)
				min_val = val;
		}

		if (min_val < cost_est)
		{
			cost_est = min_val;
			disp_est = d;
		}
	}

	spixl_map[idx].s7 = disp_est;
}



void clPhotoConsistency::do_initial_depth_estimation_host(
	vec8f *spixel_map,
	vec8u *spixel_rep,
	vec3f *cvt_img,
	cl_uint   *idx_img,
	int array_width,
	float bl_ratio,
	vector<float> &disp_levels,
	int *view_subset_mat,
	int *subset_num_per_cam,
	int start_view, int end_view
)
{
	//for (int i = 0; i < subset_num_per_cam[1]; i++)
		//std::cout << view_subset_mat[view_count + i] << ", ";

	std::cout << "Start of The Host Code" << std::endl;
	for (int z = start_view ; z < end_view ; z++)
	{
		for (int y = 0 ; y < map_size.y ; y++)
			for (int x = 0 ; x < map_size.x ; x++)
				init_depth_map(spixel_map, spixel_rep, cvt_img, idx_img, subset_num_per_cam, view_subset_mat, disp_levels.data(), x, y, z);
		std::cout << " Image " << z << " Is Done. " << std::endl;
	}
		
}



void clPhotoConsistency::img_translate(cl_uint *idx_img, cl_float8 *spixel_map, float min_disp, float max_disp, int start_view, int end_view)
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
				float d = spixel_map[map_size.s0*map_size.s1*k + sp_idx].s7;

				unsigned char d_scale = (unsigned char)floor(((d - min_disp) / (max_disp - min_disp)) * 255);
				test_cam.at<float>(i, j) = (d-10) / 90;
				test_cam_2.at<unsigned char>(i, j) = d_scale;
			}

		std::string address = "../results/1- initialize disparity/initD_dev" + std::to_string(k) + ".png";
		imwrite(address, test_cam_2);
		//imshow("My Window", test_cam);
		//waitKey(0);
	}
}
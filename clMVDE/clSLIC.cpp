#include "stdafx.h"

#include "clSLIC.h"
#include <vector>


clSLIC::clSLIC(cl::Program program, system_settings *settings_, cl_int2 img_size, cl_int2 map_size)
{
	this->program  = program;
	this->in_img = in_img;
	this->img_size = img_size;
	this->map_size = map_size;	
	this->settings = settings_;
	
	max_xy_dist = 1.0f / (1.4242f * this->settings->spixl_size);
	max_color_dist = 15.0f / (1.7321f * 128);
	max_xy_dist    *= max_xy_dist;
	max_color_dist *= max_color_dist; 

	//num_neighbor_check = 9;

	float num_pixl_to_search = (float)(this->settings->spixl_size * this->settings->spixl_size * 9);
	num_grid_per_center = (int)ceil(num_pixl_to_search / (float)(LOCAL_SIZE_UPDATE * LOCAL_SIZE_UPDATE));

	// Alloc Buffs, Trans host to device
	/**/
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	cl_int err;
	lab_img_dev = new cl::Buffer(context, CL_MEM_READ_WRITE,
		img_size.x * img_size.y * sizeof(vec3f), nullptr, &err);

	spixl_map_dev = new cl::Buffer(context, CL_MEM_READ_WRITE,
		map_size.x * map_size.y * sizeof(vec8f), nullptr, &err);
	
	idx_img_dev = new cl::Buffer(context, CL_MEM_READ_WRITE,
		img_size.x * img_size.y * sizeof(cl_uint), nullptr, &err);

	if (settings->edge_enable == true)
		edge_val_dev = new cl::Buffer(context, CL_MEM_READ_WRITE, img_size.x * img_size.y * sizeof(float), nullptr, &err);
}


clSLIC::~clSLIC()
{
	/**
	std::vector<cl::Buffer> no_need;
	//no_need.push_back(*in_img_dev);
	no_need.push_back(*idx_img_dev);
	no_need.push_back(*spixl_map_dev);
	no_need.push_back(*lab_img_dev);
	no_need.clear();
	/**/
	//std::cout << "inside the destructor !!" << std::endl;
	
	delete idx_img_dev;
	delete spixl_map_dev;
	delete lab_img_dev;

	if (settings->edge_enable == true) delete edge_val_dev;
	
}



void clSLIC::do_super_pixel_seg(vec3u *in_img, vec3f *cvt_img, vec8f *spixl_map, cl_uint *idx_img)
{
	
	// --------------------------------------------------------------------- 
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	//---------------------------------------------------------------------
	// Allocate Input Image Buffer
	cl_int err;
	this->in_img = in_img;
	in_img_dev = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		img_size.x * img_size.y * sizeof(vec3u), this->in_img, &err);

	//--------------------------------------------------------------------
	// Exe SLIC Pipeline
	cvt_color_space();
	init_cluster_centers();

	if (settings->edge_enable == true)	apply_edge_values();

	//init_labels();
	find_center_association();
	
	for (int i = 0 ; i < settings->no_iter ; i++)
	{
		update_cluster_center();
		//alternative_update_cluster();
		find_center_association();
	}

	if (settings->enforce_connectivity == true)
		enforce_connectivity();


	// ------------------------------------------------------------------
	// Update from Device to Host	
	cl::CommandQueue queue(context, devices[0]);

	this->lab_img = cvt_img;
	this->spixl_map = spixl_map;
	this->idx_img = idx_img;
	
	err = queue.enqueueReadBuffer(*lab_img_dev, CL_TRUE, 0, img_size.x * img_size.y * sizeof(vec3f), this->lab_img);
	queue.finish();
	
	err = queue.enqueueReadBuffer(*spixl_map_dev, CL_TRUE, 0, map_size.x * map_size.y * sizeof(vec8f), this->spixl_map);
	queue.finish();
	
	err = queue.enqueueReadBuffer(*idx_img_dev, CL_TRUE, 0, img_size.x * img_size.y * sizeof(cl_uint), this->idx_img);
	queue.finish();
	
	// --------------------------------------------------------------------
	// Free The Device Memory
	delete in_img_dev;
}


void clSLIC::cvt_color_space()
{
	cl_int err;

	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
	std::cout << "Device Name: " << devices[0].getInfo<CL_DEVICE_VENDOR>() << std::endl;

	// Define kernel 
	cl::Kernel kernel(program, "cvt");
	err = kernel.setArg(0, *in_img_dev);
	err = kernel.setArg(1, *lab_img_dev);
	err = kernel.setArg(2, img_size);
	if (err != CL_SUCCESS)
		std::cout << "Inside Cvt Image Function. Set Arguman Error. Error Number: " << err << std::endl;

	// Exe kernel
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float)img_size.x / LOCAL_SIZE_UPDATE) ) * LOCAL_SIZE_UPDATE;
	grid_size.y = (int)(ceil((float)img_size.y / LOCAL_SIZE_UPDATE) ) * LOCAL_SIZE_UPDATE;

	
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE_UPDATE, LOCAL_SIZE_UPDATE));
	queue.finish();

	if (err != CL_SUCCESS)	std::cout << "Inside Cvt Image Function. Kernel Call Error. Error Number: " << err << std::endl;	
}



void clSLIC::init_cluster_centers()
{
	cl_int err;
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// Define Kernel: 
	cl::Kernel kernel(program, "init_cluster_centers");
	err = kernel.setArg(0, *lab_img_dev);
	err = kernel.setArg(1, *spixl_map_dev);
	err = kernel.setArg(2, img_size);
	err = kernel.setArg(3, map_size);
	err = kernel.setArg(4, settings->spixl_size);
	if (err != CL_SUCCESS)
		std::cout << "Init_Cluster_Center: Kernel declaration error. Error no: "<< err << std::endl;

	// Config kernel
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float)map_size.x / (float)LOCAL_SIZE_UPDATE)) * LOCAL_SIZE_UPDATE;
	grid_size.y = (int)(ceil((float)map_size.y / (float)LOCAL_SIZE_UPDATE)) * LOCAL_SIZE_UPDATE;

	// Exe the kernel
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE_UPDATE, LOCAL_SIZE_UPDATE));
	queue.finish();

	if (CL_SUCCESS)	std::cout << " Init_Cluster_Center: Kernel launch error. Error no: " << err << std::endl;
}


void clSLIC::apply_edge_values()
{
	cl_int err;
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// ---------------------------------
	// Define Kernel
	cl::Kernel kernel(program, "edge_compute_alternative");
	err = kernel.setArg(0, *lab_img_dev);
	err = kernel.setArg(1, *edge_val_dev);
	err = kernel.setArg(2, img_size);
	if (err != CL_SUCCESS)
		printText("Edge Magnitiude Computation Error: Set Arguman Error. Error No: " + to_string(err));

	// --------------------------------
	// Config the kernel
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float)img_size.x / (float)LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)img_size.y / (float)LOCAL_SIZE)) * LOCAL_SIZE;

	// Exe Kernel
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE));
	queue.finish();
	if (err != CL_SUCCESS) printText("Execution Error: Apply_edge_values First kernel. Error No: " + to_string(err));

	// -------------------------------
	// Define Kernel 2
	cl::Kernel kernel2(program, "apply_edge_alternative");
	err = kernel2.setArg(0, *lab_img_dev);
	err = kernel2.setArg(1, *edge_val_dev);
	err = kernel2.setArg(2, *spixl_map_dev);
	err = kernel2.setArg(3, this->img_size);
	err = kernel2.setArg(4, this->map_size);
	if (err != CL_SUCCESS) printText("SET ARGUMAN ERROR: Apply Edge KERNEL 2. ERROR NO: " + to_string(err));

	// -------------------------------
	// Config the Kernel 2
	grid_size.x = (int)(ceil((float)map_size.x / (float)LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)map_size.y / (float)LOCAL_SIZE)) * LOCAL_SIZE;

	// ------------------------------
	// Exe Kernel 2
	err = queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE));
	queue.finish();
	if (err != CL_SUCCESS) printText("Execution Error: APPLY_EDGE_VALUE SECOND KERNEL. Error No: " + to_string(err));
}



void clSLIC::init_labels()
{
	cl_int err;
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// ------------------------------
	// Define Kernel
	cl::Kernel kernel(program, "init_label_per_pixl");
	err = kernel.setArg(0, *idx_img_dev);
	err = kernel.setArg(1, img_size);
	err = kernel.setArg(2, map_size);
	err = kernel.setArg(3, settings->spixl_size);
	if (err != CL_SUCCESS) printText("SET ARGUMAN ERROR: INIT_LABEL FUNCTION. ERROR NO: " + to_string(err));

	// -----------------------------
	// Configure Kernel
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float)img_size.x / (float)LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)img_size.y / (float)LOCAL_SIZE)) * LOCAL_SIZE;

	// Exe Kernel
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, NULL, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE));
	queue.finish();
	if (err != CL_SUCCESS) printText("EXE ERROR: INIT_LABEL FUNCTION. ERROR NO: " + to_string(err));
}




void clSLIC::find_center_association()
{
	cl_int err;
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// Define Kernel
	cl::Kernel kernel(program, "find_center_association");
	err = kernel.setArg(0, *lab_img_dev);
	err = kernel.setArg(1, *spixl_map_dev);
	err = kernel.setArg(2, *idx_img_dev);
	err = kernel.setArg(3, img_size);
	err = kernel.setArg(4, map_size);
	err = kernel.setArg(5, settings->spixl_size);
	err = kernel.setArg(6, max_xy_dist);
	err = kernel.setArg(7, max_color_dist);
	err = kernel.setArg(8, settings->slic_color_weight);

	if (err != CL_SUCCESS)	std::cout << "Find_Center_Association: kernel declaration. Error no: " << err << std::endl;

	// Exe kernel
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float)img_size.x / LOCAL_SIZE_UPDATE)) * LOCAL_SIZE_UPDATE;
	grid_size.y = (int)(ceil((float)img_size.y / LOCAL_SIZE_UPDATE)) * LOCAL_SIZE_UPDATE;

	cl::CommandQueue queue(context, devices[0]);

	auto start_time = std::chrono::high_resolution_clock::now();	// Take the start time
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE_UPDATE, LOCAL_SIZE_UPDATE));
	queue.finish();
	auto end_time = std::chrono::high_resolution_clock::now();	// Take the end time
	auto duration = duration_cast<microseconds>(end_time - start_time);
	std::cout << "Time of find center = " << duration.count() << std::endl;

	if (err != CL_SUCCESS)	std::cout << "Find_Center_Association: run kernel. Error no: " << err << std::endl;
}



void clSLIC::update_cluster_center()
{	
	cl_int err;
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	cl_int cluster_per_line =  settings->spixl_size * 3 / LOCAL_SIZE_UPDATE;

	cl::Buffer accum_map_dev(context, CL_MEM_READ_WRITE,
			map_size.x * num_grid_per_center * map_size.y * sizeof(cl_float8), nullptr, &err);

	// Config Kernel 1
	cl_int3 grid_size;
	grid_size.x = map_size.x * LOCAL_SIZE_UPDATE;
	grid_size.y = map_size.y * LOCAL_SIZE_UPDATE;
	grid_size.z = num_grid_per_center;

	// Define Kernel 1
	cl::Kernel kernel_1(program, "update_cluster_center");
	err = kernel_1.setArg(0, *lab_img_dev);
	err = kernel_1.setArg(1, *idx_img_dev);
	err = kernel_1.setArg(2, accum_map_dev);
	err = kernel_1.setArg(3, map_size);
	err = kernel_1.setArg(4, img_size);
	err = kernel_1.setArg(5, settings->spixl_size);
	err = kernel_1.setArg(6, cluster_per_line);
	err = kernel_1.setArg(7, LOCAL_SIZE_UPDATE * LOCAL_SIZE_UPDATE * sizeof(cl_float3), nullptr);
	err = kernel_1.setArg(8, LOCAL_SIZE_UPDATE * LOCAL_SIZE_UPDATE * sizeof(cl_float2), nullptr);
	err = kernel_1.setArg(9, LOCAL_SIZE_UPDATE * LOCAL_SIZE_UPDATE * sizeof(cl_float), nullptr);
	err = kernel_1.setArg(10, sizeof(bool), nullptr);
	if (err != CL_SUCCESS)
		std::cout << "Update_Cluster_Center: kernel declaration. Error no: " << err << std::endl;

	// Create Queue 
	cl::CommandQueue queue(context, devices[0]);

	// Exe kernel 1
	err = queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y, grid_size.z), cl::NDRange(LOCAL_SIZE_UPDATE, LOCAL_SIZE_UPDATE, 1));
	queue.finish();
	if (err != CL_SUCCESS)	std::cout << "Update_Cluster_Center: Kernel launch. Error no: " << err << std::endl;

	/**/
	// Define Kernel 2
	cl::Kernel kernel_2(program, "finalize_reduction_result");
	err = kernel_2.setArg(0, accum_map_dev);
	err = kernel_2.setArg(1, *spixl_map_dev);
	err = kernel_2.setArg(2, map_size);
	err = kernel_2.setArg(3, num_grid_per_center);
	if (err != CL_SUCCESS)
		std::cout << "Inside Update Cluster Center Function. Set Arguman2 Error. Error Number: " << err << std::endl;

	// Exe kernel 2
	cl_int2 grid_size2;
	grid_size2.x = (int)(ceil((float)map_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size2.y = (int)(ceil((float)map_size.y / LOCAL_SIZE)) * LOCAL_SIZE;

	err = queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(grid_size2.x, grid_size2.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE));
	queue.finish();
	 
	if (err != CL_SUCCESS)	std::cout << " Update Cluster Center: Kernel_2 Launch Error. Error no: " << err << std::endl;

	/**/

}


void clSLIC::enforce_connectivity()
{
	cl_int err;
	std::string err_msg = "enforce connectivity";
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// Allocate Buffers
	cl::Buffer tmp_idx_img(context, CL_MEM_READ_WRITE, img_size.x * img_size.y * sizeof(cl_uint), nullptr, &err);
	errorHandler(buff_alloc, err, err_msg);

	// Configure Kernel
	cl_int2 grid_size;
	grid_size.x = (int)(ceil((float)img_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)img_size.y / LOCAL_SIZE)) * LOCAL_SIZE;

	// Define Kernel 
	cl::Kernel kernel_1(program, "supress_local_lable", &err);
	err = kernel_1.setArg(0, *idx_img_dev);
	err = kernel_1.setArg(1, tmp_idx_img);
	err = kernel_1.setArg(2, img_size);
	errorHandler(kernel_def, err, err_msg);

	// Exe Kernel 1
	cl::CommandQueue queue(context, devices[0]); 
	err = queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE)); queue.finish();
	errorHandler(kernel_exe, err, err_msg);
	
	// Define Kernel 2
	cl::Kernel kernel_2(program, "supress_local_lable", &err);
	err = kernel_2.setArg(0, tmp_idx_img);
	err = kernel_2.setArg(1, *idx_img_dev);
	err = kernel_2.setArg(2, img_size);
	errorHandler(kernel_def, err, err_msg + " kernel 2");

	// Exe Kernel 2
	err = queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE)); queue.finish();
	errorHandler(kernel_exe, err, err_msg + " kernel 2");
}


void clSLIC::alternative_update_cluster()
{
	cl_int err;
	cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
	
	// ---------------------------------------------
	// Define Kernel
	cl::Kernel kernel(program, "alternate_update_cluster", &err);
	err = kernel.setArg(0, *spixl_map_dev);
	err = kernel.setArg(1, *lab_img_dev);
	err = kernel.setArg(2, *idx_img_dev);
	err = kernel.setArg(3, img_size);
	err = kernel.setArg(4, map_size);
	err = kernel.setArg(5, settings->spixl_size);

	errorHandler(kernel_def, err, "Location of Error: alternative update function.");

	// --------------------------------------------
	// Kernel Configuration
	vec2i grid_size;
	grid_size.x = (int)(ceil((float)map_size.x / LOCAL_SIZE)) * LOCAL_SIZE;
	grid_size.y = (int)(ceil((float)map_size.y / LOCAL_SIZE)) * LOCAL_SIZE;

	// --------------------------------------------
	// Exe Kernel
	cl::CommandQueue queue(context, devices[0]);
	err = queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(grid_size.x, grid_size.y), cl::NDRange(LOCAL_SIZE, LOCAL_SIZE));
	queue.finish();
	errorHandler(kernel_exe, err, "Location of Error: alternative update function");

}

void clSLIC::draw_segmentation_lines(cl_uchar3 *in_img, cl_uchar3 *out_img)
{

	int height = img_size.y;
	int width = img_size.x;

	for (int i = 1 ; i < height - 1 ; i++)
	{
		for (int j = 1 ; j < width - 1 ; j++)
		{

			if (idx_img[i*width + j] != idx_img[i*width + j + 1] ||
				idx_img[i*width + j] != idx_img[i*width + j - 1] ||
				idx_img[i*width + j] != idx_img[(i - 1)*width + j] ||
				idx_img[i*width + j] != idx_img[(i + 1)*width + j])
			{
				out_img[i*width + j].s0 = 0;
				out_img[i*width + j].s1 = 0;
				out_img[i*width + j].s2 = 255;
			}
			else
			{
				out_img[i*width + j].x = in_img[i*width + j].x;
				out_img[i*width + j].y = in_img[i*width + j].y;
				out_img[i*width + j].z = in_img[i*width + j].z;
			}

		}
	}


}
#ifndef HEADERS

#define HEADERS

#include <CL/cl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <math.h>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

typedef cl_float8 vec8f;
typedef cl_float4 vec4f;
typedef cl_float3 vec3f;
typedef cl_float2 vec2f;

typedef cl_int3 vec3i;
typedef cl_int2 vec2i;

typedef cl_uchar8 vec8u;
typedef cl_uchar3 vec3u;

enum CL_Error_TYPE{buff_alloc = 0, kernel_def = 1, kernel_exe = 2, read_mem = 3};

#define LOCAL_SIZE 16
#define LOCAL_SIZE_UPDATE 16

void printText(std::string txt);
void printError(cl_int err_no, std::string message);
void errorHandler(CL_Error_TYPE type, cl_int err, std::string msg);

vec2i makeVec2i(int x, int y);
vec3i makeVec3i(int x, int y, int z);

vec2f makeVec2f(float x, float y);
vec3f makeVec3f(float x, float y, float z);
vec8f makeVec8f(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7);

float euDistance3D(vec3f d1, vec3f d2);
vec3f crossVec3f(vec3f a, vec3f b);
vec3f normalizeVec3f(vec3f a);

struct system_settings {
	int spixl_size;
	float slic_color_weight;
	int array_width, array_height;
	int no_iter;
	bool enforce_connectivity = false;
	bool edge_enable = false;

	int num_disp_levels;
	int neib_hor;
	int neib_ver;
	int min_disp;
	int max_disp;
	int inc;
	float bl_ratio;

	int kernel_size;
	int kernel_step;
	float fuse;
	float gamma;
	float alpha;
	int no_prop;
};



#endif // !HEADERS
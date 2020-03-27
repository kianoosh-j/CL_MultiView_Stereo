#include "stdafx.h"

#include "header.h"


void loadImageIn(Mat &in_cam, vec3u *in_img, int height, int width)
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			in_img[i*width + j].s0 = in_cam.at<Vec3b>(i, j)[2];	// red
			in_img[i*width + j].s1 = in_cam.at<Vec3b>(i, j)[1];	// green
			in_img[i*width + j].s2 = in_cam.at<Vec3b>(i, j)[0]; // blue 
		}
}

void loadImageOut(Mat &out_cam, cl_uchar3 *out_img, int height, int width)
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			out_cam.at<Vec3b>(i, j)[2] = out_img[i*width + j].s0; // red
			out_cam.at<Vec3b>(i, j)[1] = out_img[i*width + j].s1; // green
			out_cam.at<Vec3b>(i, j)[0] = out_img[i*width + j].s2; // blue
		}
}



bool read_image_array(vector<Mat> &img_array, std::string img_file_addr, int num_view)
{
	bool err = false;

	vector<string> img_files;
	ifstream info_file(img_file_addr);
	string line;

	if (info_file.is_open())
	{
		while (getline(info_file, line))
			img_files.push_back(line);
	}
	else
		return true;// err = true;

	for (int i = 0; i < num_view; i++)
	{
		auto img_cam = imread(img_files[i]);

		//Size s(640, 480);
		//Mat std_img;
		//resize(img_cam, std_img, s);
		img_array.push_back(img_cam);
	}

	return err;
}



void imgSave(cv::Mat img, int view_no)
{
	std::string address = "../results/slic output/green_new_2" + std::to_string(view_no) + ".png";
	imwrite(address, img);
}



void show_img(vec3u *res_img, int height, int width, int view_num)
{
	//namedWindow("My Window", WINDOW_AUTOSIZE);

	Mat out_cam(height, width, CV_8UC3);

	for (int i = 0; i < view_num; i++)
	{
		auto res_img_start = res_img + i*width*height;
		loadImageOut(out_cam, res_img_start, height, width);

		//resize(out_cam, out_cam, Size(800, 800));
		//imshow("My Window", out_cam);
		imgSave(out_cam, i);
		//waitKey(0);
	}
}

void printText(std::string txt)
{
	std::cout << txt << std::endl;
}

void printError(cl_int err_no, std::string message)
{
	std::cout << "Error: " + message + ". Error No: " + std::to_string(err_no) << std::endl;
}

void errorHandler(CL_Error_TYPE type, cl_int err, std::string msg)
{
	if (err != CL_SUCCESS)
	{
		switch(type)
		{ 
			case 0:
				std::cout << "Error No " << to_string(err) << ": Buffer Allocate Error." + msg<< std::endl;
			case 1:
				std::cout << "Error No " << to_string(err) << ": Kernel Define Error." + msg << std::endl;
			case 2:
				std::cout << "Error No " << to_string(err) << ": Kernel Exe Error." + msg << std::endl;
			case 3:
				std::cout << "Error No " << to_string(err) << ": Read From Device to Host Error." + msg << std::endl;
		}
	}
}


vec2f makeVec2f(float x, float y)
{
	vec2f tmp;
	tmp.x = x; tmp.y = y;
	return tmp;
}

vec3f makeVec3f(float x, float y, float z)
{
	vec3f tmp;
	tmp.x = x; tmp.y = y; tmp.z = z;
	return tmp;
}


vec2i makeVec2i(int x, int y)
{
	vec2i tmp;
	tmp.x = x; tmp.y = y;
	return tmp;
}

vec3i makeVec3i(int x, int y, int z)
{
	vec3i tmp;
	tmp.x = x; tmp.y = y; tmp.z = z;
	return tmp;
}

vec8f makeVec8f(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7)
{
	vec8f tmp;
	tmp.s0 = s0; tmp.s1 = s1; tmp.s2 = s2; tmp.s3 = s3; tmp.s4 = s4; tmp.s5 = s5; tmp.s6 = s6; tmp.s7 = s7;
	return tmp;
}


float euDistance3D(vec3f d1, vec3f d2)
{
	float dist = 0.0;
	dist += pow((d1.x - d2.x), 2);
	dist += pow((d1.y - d2.y), 2);
	dist += pow((d1.z - d2.z), 2);

	return sqrt(dist);
}


vec3f crossVec3f(vec3f a, vec3f b)
{
	vec3f c;
	c.x = a.x * b.z - a.z * b.y;
	c.y = b.x * a.z - a.x * b.z;
	c.z = a.x * b.y - a.y * b.x;

	return c;
}


vec3f normalizeVec3f(vec3f a)
{
	float norm = sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2) );
	vec3f a_normal = makeVec3f(a.x / norm, a.y / norm, a.z / norm);

	return a_normal;
}


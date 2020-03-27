#include "header.h"

void loadImageIn(Mat &in_cam, vec3u *in_img, int height, int width);
void loadImageOut(Mat &out_cam, cl_uchar3 *out_img, int height, int width);
bool read_image_array(vector<Mat> &img_array, std::string img_file_addr, int num_view);
void show_img(vec3u *res_img, int height, int width, int view_num);
void imgSave(cv::Mat img, int view_no);
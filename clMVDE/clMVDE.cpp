// clMVDE.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "header.h"
#include "pipeline.h"




int main()
{
	system_settings *s = new system_settings;
	s->spixl_size = 8;
	s->slic_color_weight = 0.6;
	s->array_height = 3;
	s->array_width = 3;
	s->no_iter = 5;
	s->enforce_connectivity = false;
	s->edge_enable =  false;

	s->num_disp_levels = 30;
	s->neib_hor = 1;
	s->neib_ver = 1;
	s->min_disp = 30;// 10;
	s->max_disp = 60;// 100;
	s->inc = 1;
	s->bl_ratio = 1.03590;// 0.625;

	s->kernel_size = 1080;
	s->kernel_step = 13;
	s->fuse = 1;
	s->gamma = 2;
	s->alpha = 6;
	s->no_prop = 5;

	pipeline *model = new pipeline("data.txt", s);
	model->exe_pipeline();

	system("pause");
    return 0;	
}
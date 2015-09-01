#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "device_functions.h"


using namespace cv;
using namespace cv::cuda;

#define SHARED_SIZE_H 10
#define SHARED_SIZE_W 34
#define SHARED_SIZE 340


__device__ int kernal[9] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };

//自定义核函数
__global__ void lapalace_kernal(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= src.cols || y >= src.rows)
		return;

	/*
	int dst_blue = src(y-1,x).x*kernal[1] +
		src(y,x-1).x*kernal[3] +
		src(y, x).x*kernal[4] +
		src(y, x+1).x*kernal[5] +
		src(y + 1, x).x*kernal[7];

	dst_blue = dst_blue > 255 ? 255 : dst_blue;
	dst_blue = dst_blue < 0 ? 0 : dst_blue;

	int dst_green = src(y - 1, x).y*kernal[1] +
		src(y, x - 1).y*kernal[3] +
		src(y, x).y*kernal[4] +
		src(y, x + 1).y*kernal[5] +
		src(y + 1, x).y*kernal[7];
	dst_green = dst_green > 255 ? 255 : dst_green;
	dst_green = dst_green < 0 ? 0 : dst_green;

	int dst_red = src(y - 1, x).z*kernal[1] +
		src(y, x - 1).z*kernal[3] +
		src(y, x).z*kernal[4] +
		src(y, x + 1).z*kernal[5] +
		src(y + 1, x).z*kernal[7];
	dst_red = dst_red > 255 ? 255 : dst_red;
	dst_red = dst_red < 0 ? 0 : dst_red;


	dst(y, x) = make_uchar3(dst_blue, dst_green, dst_red);
	*/
	
	
	//共享内存版本
	//copy data to shared memory
	int shared_index_x = threadIdx.x + 1;
	int shared_index_y = threadIdx.y + 1;
	
	__shared__ uchar3 s_data[SHARED_SIZE];
	s_data[shared_index_y*SHARED_SIZE_W + shared_index_x] = src(y, x);
	uchar3 center = s_data[shared_index_y*SHARED_SIZE_W + shared_index_x];
	
	if (shared_index_x == 1)
	{
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] = center;
		if (shared_index_y == 1)
			s_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x - 1] = center;
		if (shared_index_y == SHARED_SIZE_H - 2)
			s_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x - 1] = center;
	}
	if (shared_index_x == SHARED_SIZE_W - 2)
	{
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] = center;
		if (shared_index_y == 1)
			s_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x + 1] = center;
		if (shared_index_y == SHARED_SIZE_H - 2)
			s_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x + 1] = center;
	}
	if (shared_index_y == 1)
	{
		s_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] = center;
	}
	if (shared_index_y == SHARED_SIZE_H - 2)
	{
		s_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x] = center;
	}
	__syncthreads();
		
	int dst_blue = center.x * 5 -
		s_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x].x -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1].x -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1].x -
		s_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x].x;


	int dst_green = center.y * 5 -
		s_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x].y -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1].y -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1].y -
		s_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x].y;


	int dst_red = center.z * 5 -
		s_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x].z -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1].z -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1].z -
		s_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x].z;
	
	center.x = dst_blue > 255 ? 255 : dst_blue < 0 ? 0 : dst_blue;
	center.y = dst_green > 255 ? 255 : dst_green < 0 ? 0 : dst_green;
	center.z = dst_red > 255 ? 255 : dst_red < 0 ? 0 : dst_red;
	dst(y, x) = center;

	
}

extern "C"
void lapalace(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst)
{
	dim3 block(32, 8);
	dim3 grid((src.cols+block.x-1)/block.x, (src.rows+block.x-1)/block.y);
	//cudaSafeCall(cudaFuncSetCacheConfig(lapalace_kernal, cudaFuncCachePreferShared));
	lapalace_kernal << <grid, block >> >(src, dst);
	//cudaSafeCall(cudaGetLastError());
	// sync host and stop computation timer
	cudaSafeCall(cudaDeviceSynchronize());
}



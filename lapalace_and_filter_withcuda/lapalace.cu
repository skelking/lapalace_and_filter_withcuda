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

__shared__ int b_data[SHARED_SIZE];
__shared__ int g_data[SHARED_SIZE];
__shared__ int r_data[SHARED_SIZE];


//自定义核函数
__global__ void lapalace_kernal(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	
	//共享内存版本
	//copy data to shared memory
	int shared_index_x = threadIdx.x + 1;
	int shared_index_y = threadIdx.y + 1;

	
	uchar3 center = src(y,x);
	int blue = center.x;
	int green = center.y;
	int red = center.z;

	b_data[shared_index_y*SHARED_SIZE_W + shared_index_x] = blue;
	g_data[shared_index_y*SHARED_SIZE_W + shared_index_x] = green;
	r_data[shared_index_y*SHARED_SIZE_W + shared_index_x] = red;

	if (shared_index_x == 1)
	{
		b_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] = blue;
		g_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] = green;
		r_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] = red;
	}
	if (shared_index_x == SHARED_SIZE_W - 2)
	{
		b_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] = blue;
		g_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] = green;
		r_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] = red;
	}
	if (shared_index_y == 1)
	{	
		b_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] = blue;
		g_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] = green;
		r_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] = red;
	}
	if (shared_index_y == SHARED_SIZE_H - 2)
	{
		b_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x] = blue;
		g_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x] = green;
		r_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x] = red;
	}
	__syncthreads();

	int dst_blue = blue * 5 -
		b_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] -
		b_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] -
		b_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] -
		b_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x];

	int dst_green = green * 5 -
		g_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] -
		g_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] -
		g_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] -
		g_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x];

	int dst_red = red * 5 -
		r_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] -
		r_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] -
		r_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] -
		r_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x];
	
	
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



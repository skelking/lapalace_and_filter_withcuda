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
	int z = threadIdx.z;

	if (x >= src.cols || y >= src.rows)
		return;

	//共享内存版本
	//copy data to shared memory
	int shared_index_x = threadIdx.x + 1;
	int shared_index_y = threadIdx.y + 1;

	__shared__ uchar s_data[SHARED_SIZE];
	uchar center;
	if (z == 0)
		center = src(y, x).x;
	else if (z == 1)
		center = src(y, x).y;
	else if (z == 2)
		center = src(y, x).z;

	s_data[shared_index_y*SHARED_SIZE_W + shared_index_x] = center;


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


	int dst_value = center * 5 -
		s_data[(shared_index_y - 1)*SHARED_SIZE_W + shared_index_x] -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x - 1] -
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x + 1] -
		s_data[(shared_index_y + 1)*SHARED_SIZE_W + shared_index_x];


	

	dst_value = dst_value > 255 ? 255 : dst_value < 0 ? 0 : dst_value;
	if (z == 0)
		dst(y, x).x = dst_value;
	else if (z ==1)
		dst(y, x).y = dst_value;
	else if (z == 2)
		dst(y, x).z = dst_value;

}

extern "C"
void lapalace(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst)
{
	dim3 block(32, 8,3);
	dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.x - 1) / block.y, 3);
	//cudaSafeCall(cudaFuncSetCacheConfig(lapalace_kernal, cudaFuncCachePreferShared));
	lapalace_kernal << <grid, block >> >(src, dst);
	//cudaSafeCall(cudaGetLastError());
	// sync host and stop computation timer
	cudaSafeCall(cudaDeviceSynchronize());
}



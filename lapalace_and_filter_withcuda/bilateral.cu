#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "device_functions.h"

using namespace cv;
using namespace cv::cuda;


__device__ __forceinline__ float norm_l1(const float3& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z); }
__device__ float3 uchar3Tofloat3(uchar3  a)
{
	float3 dst;
	dst.x = (a.x);
	dst.y = (a.y);
	dst.z = (a.z);
	return dst;
}

__device__ uchar3 float3Touchar3(float3 a)
{
	uchar3 dst;
	dst.x = (a.x);
	dst.y = (a.y);
	dst.z = (a.z);
	return dst;
}

__device__ float3 float3sub(float3 a, float3 b)
{
	float3 dst;
	dst.x = (a.x - b.x);
	dst.y = (a.y - b.y);
	dst.z = (a.z - b.z);
	return dst;
}

__global__ void bilateral_kernal(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst, const int ksz, const float sigma_spatial2_inv_half, const float sigma_color2_inv_half)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= src.cols || y >= src.rows)
		return;

	int r = ksz / 2;
	__shared__ uchar3 s_data[2000];
	int shared_index_x = threadIdx.x + r;
	int shared_index_y = threadIdx.y + r;
	int SHARED_SIZE_W = blockDim.x + 2 * r;
	int SHARED_SIZE_H = blockDim.y + 2 * r;

	s_data[shared_index_x + shared_index_y*SHARED_SIZE_W] = src(y, x);
	
	if (shared_index_x < 2 * r)
	{
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x - r] =src(y, x - r);
	}
	if (shared_index_x > SHARED_SIZE_W - 2 * r - 1)
	{
		s_data[shared_index_y*SHARED_SIZE_W + shared_index_x + r] = src(y, x + r);
	}
	if (shared_index_y < 2 * r)
	{
		s_data[(shared_index_y - r)*SHARED_SIZE_W + shared_index_x] = src(y - r, x);
	}
	if (shared_index_y > SHARED_SIZE_H - 2 * r - 1)
	{
		s_data[(shared_index_y + r)*SHARED_SIZE_W + shared_index_x] = src(y + r, x);
	}

	
	float3 center = uchar3Tofloat3( s_data[shared_index_y*SHARED_SIZE_W + shared_index_x]);
	float3 sum1 = make_float3(0, 0, 0);
	float sum2 = 0;
	int tx = shared_index_x - r + ksz;
	int ty = shared_index_y - r + ksz;

	for (int cy = shared_index_y - r; cy < ty; ++cy)
		for (int cx = shared_index_x - r; cx < tx; ++cx)
		{
			float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
			

			float3 value = uchar3Tofloat3(s_data[cy*SHARED_SIZE_W + cx]);

			float weight = ::exp(space2 * sigma_spatial2_inv_half + norm_l1(float3sub(value, center))*norm_l1(float3sub(value, center)) * sigma_color2_inv_half);
			
			sum1.x = (weight*value.x+sum1.x);
			sum1.y = (weight*value.y+sum1.y);
			sum1.z = (weight*value.z+sum1.z);

			sum2 = sum2 + weight;
		}
	sum1.x = (sum1.x/sum2);
	sum1.y = (sum1.y / sum2);
	sum1.z = (sum1.z / sum2);

	dst(y, x) = float3Touchar3(sum1);
	
	dst(y, x) = src(y, x);
}

extern "C"
void bilateral(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst, const int ksz, const float sigma_spatial, const float sigma_color)
{
	dim3 block(32, 8);
	dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.x - 1) / block.y);
	float sigma_spatial2_inv_half = -0.5f / (sigma_spatial * sigma_spatial);
	float sigma_color2_inv_half = -0.5f / (sigma_color * sigma_color);

	bilateral_kernal << <grid, block >> >(src, dst, ksz, sigma_spatial2_inv_half, sigma_color2_inv_half);
	cudaSafeCall(cudaDeviceSynchronize());
}
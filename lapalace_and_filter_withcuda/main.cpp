#include <opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudacodec.hpp>
#include <cuda_runtime.h>


using namespace cv;
using namespace cv::cuda;

extern "C"
void lapalace(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst);


Stream stream;
int main()
{
	int num_device = cuda::getCudaEnabledDeviceCount();
	Mat source = imread("FOV00002-28.bmp");
	
	if (!source.data)
		return 0;
	
	GpuMat g_source;
	
	g_source.upload(source,stream);
	
	GpuMat lapalaceImage;
	lapalaceImage.create(g_source.size(), g_source.type());
	
	GpuMat bilateralImage;
	bilateralImage.create(g_source.size(), g_source.type());
	//Í¼ÏñÈñ»¯
	
	double start_time = (double)getTickCount();
	lapalace(g_source, lapalaceImage);
	double during = ((double)getTickCount() - start_time) / getTickFrequency();

	cv::cuda::bilateralFilter(lapalaceImage, bilateralImage, 9, 40.0, 2.0, 4);

	Mat dst_image;
	bilateralImage.download(dst_image);
	
	imwrite("bilateralImage.bmp", dst_image);

	return 1;

}
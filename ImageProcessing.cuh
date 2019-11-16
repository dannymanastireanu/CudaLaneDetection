#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>


using namespace cv;
using namespace std;


#define LIMIT 255


// Device function
__global__ void convertToGrayscale(unsigned char *gray, unsigned char *r, unsigned char *g, unsigned char *b, int dimension);
__global__ void getHistrogram(unsigned int *histogram, unsigned char *image, int dimension);
__global__ void getNormalizedHistogram(double *norm_histogram, unsigned int *histogram, int dimension);
__global__ void histogramEqualization(unsigned char *eq_image, unsigned char *image, double *cumulative_sum, int dimension);
__global__ void exclusiveScanGPU(int *d_array, int *d_result, int N, int *d_aux);


// Host function
cudaError_t bgrToGrayscale(unsigned char *gray, Mat image_rgb, unsigned int size);
cudaError_t getHistogramN(double *norm_histogram, unsigned int *histogram, unsigned char *grayScaleImage, int size);
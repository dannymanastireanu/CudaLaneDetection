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
__global__ void exclusiveScanGPU(double *d_array, double *d_result, int N, double *d_aux);
__global__ void sobelFilter(unsigned char *image, unsigned char *filtered_image, int height, int width);


// Host function
cudaError_t bgrToGrayscale(unsigned char *gray, Mat image_rgb, unsigned int size);
cudaError_t getHistogramN(double *cumulativeSumHistogram, double *norm_histogram, unsigned int *histogram, unsigned char *grayScaleImage, int size);
cudaError_t doHistogramEqualization(unsigned char *eq_image, unsigned char *image, double *cumulative_sum, int dimension);
cudaError_t applySobelFilter(unsigned char *image, unsigned char *filtered_image, int width, int height);
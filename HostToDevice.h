#ifndef __HOSTTODEVICE__
#define __HOSTTODEVICE__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>

#include "ImageProcessing.cuh"

using namespace cv;
using namespace std;

cudaError_t bgr_to_grayscale(unsigned char *gray, const unsigned char *red, const unsigned char *green, const unsigned char *blue, unsigned int size);

#endif

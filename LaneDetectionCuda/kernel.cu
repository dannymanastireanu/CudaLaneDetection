#include "ImageProcessing.cuh"


int main() {

	Mat image_rgb = imread("Images/road.jpg");
	int w = image_rgb.cols;
	int h = image_rgb.rows;

	unsigned char *gray = new unsigned char[w*h];
	cudaError_t cudaStatus = bgrToGrayscale(gray, image_rgb, w*h);

	unsigned int *histrogram = new unsigned int[LIMIT + 1];
	double *normHistogram = new double[LIMIT + 1];
	double *cumulativeSumHistogram = new double[LIMIT + 1];
	cudaStatus = getHistogramN(cumulativeSumHistogram, normHistogram, histrogram, gray, w*h);


	unsigned char *eq_gray = new unsigned char[w*h];
	cudaStatus = doHistogramEqualization(eq_gray, gray, cumulativeSumHistogram, w*h);


	unsigned char *sobel_filter = new unsigned char[w*h];
	cudaStatus = applySobelFilter(eq_gray, sobel_filter, w, h);


	Mat gray_scale(h, w, CV_8U, gray);
	Mat gray_scale_eq(h, w, CV_8U, sobel_filter);
	imshow("frame", gray_scale);
	imshow("eq", gray_scale_eq);
	waitKey(0);


    return 0;
}
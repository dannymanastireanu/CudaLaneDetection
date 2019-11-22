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

	//	apply gaussian filter
	unsigned char *gaussian_image = new unsigned char[w*h];
	cudaStatus = applyGaussianFilter(gray, gaussian_image, w, h, 5);


	//	eq histogram on filtered gaus.
	unsigned char *eq_gray = new unsigned char[w*h];
	cudaStatus = doHistogramEqualization(eq_gray, gaussian_image, cumulativeSumHistogram, w*h);


	unsigned char *thresh = new unsigned char[w*h];
	cudaStatus = applyBinaryThreshold(eq_gray, thresh, w, h, 178);


	//	apply edge detection
	unsigned char *sobel_filter = new unsigned char[w*h];
	cudaStatus = applySobelFilter(thresh, sobel_filter, w, h);

	//	apply threshold


	//	put a mask on roi and get an image with just 2 lines


	//	math calculation to get numbers from this 2 lines, more exactly coordinates 



	Mat original_image(h, w, CV_8U, gray);
	Mat gauss(h, w, CV_8U, sobel_filter);
	imshow("frame", original_image);
	imshow("gauss", gauss);
	waitKey(0);


    return 0;
}
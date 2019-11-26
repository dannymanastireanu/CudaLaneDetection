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


	//	apply threshold
	unsigned char *thresh = new unsigned char[w*h];
	cudaStatus = applyBinaryThreshold(eq_gray, thresh, w, h, 178);


	//	apply edge detection
	unsigned char *sobel_filter = new unsigned char[w*h];
	cudaStatus = applySobelFilter(thresh, sobel_filter, w, h);

	//	put a mask on roi and get an image with just road lane
	unsigned char *roi = new unsigned char[w*h];
	cudaStatus = extractROI(sobel_filter, roi, w, h);


	Mat original_image(h, w, CV_8U, gray);
	Mat region_of_interest(h, w, CV_8U, roi);

	//	math calculation to get numbers from this 2 lines, more exactly coordinates
	vector<Vec4i> lines;
	HoughLinesP(region_of_interest, lines, 1, CV_PI / 180, 20, 20, 30);

	plotLines(lines, image_rgb);

	delete[] gray;
	delete[] histrogram;
	delete[] normHistogram;
	delete[] cumulativeSumHistogram;
	delete[] gaussian_image;
	delete[] eq_gray;
	delete[] thresh;
	delete[] sobel_filter;
	delete[] roi;

    return 0;
}
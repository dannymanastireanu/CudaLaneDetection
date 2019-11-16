#include "ImageProcessing.cuh"




int main() {

	Mat image_rgb = imread("Images/road.jpg");
	int w = image_rgb.cols;
	int h = image_rgb.rows;

	unsigned char *gray = new unsigned char[w*h];
	cudaError_t cudaStatus = bgrToGrayscale(gray, image_rgb, w*h);

	unsigned int *histrogram = new unsigned int[LIMIT + 1];
	double* normHistogram = new double[LIMIT + 1];
	cudaStatus = getHistogramN(normHistogram, histrogram, gray, w*h);


	for (int i = 0; i < LIMIT + 1; ++i)
		printf("%lf", normHistogram[i]);


	/*	Eq. Histogram	*/
	//
	//unsigned int *d_histogram;

	//double *h_norm_histogram = new double[LIMIT + 1];
	//double *d_norm_histogram;

	//double *d_cumulative_sum;
	//double *h_cumulative_sum = new double[LIMIT + 1];

	//// init histogram on device with 0
	//for (int i = 0; i < LIMIT + 1; ++i)
	//	h_histrogram[i] = 0;

	//int bytes_histogram = (LIMIT + 1) * sizeof(unsigned int);
	//int bytes_norm_histogram = (LIMIT + 1) * sizeof(double);
	//
	//cudaMalloc(&d_histogram, bytes_histogram);
	//cudaMemcpy(d_histogram, h_histrogram, bytes_histogram, cudaMemcpyHostToDevice);

	//cudaMalloc(&d_norm_histogram, bytes_norm_histogram);

	//getHistrogram << <no_blocks, no_threads >> > (d_histogram, d_gray, w*h);
	//cudaMemcpy(h_histrogram, d_histogram, bytes_histogram, cudaMemcpyDeviceToHost);

	//getNormalizedHistogram << <1, 256 >> > (d_norm_histogram, d_histogram, w*h);
	//cudaMemcpy(h_norm_histogram, d_norm_histogram, bytes_norm_histogram, cudaMemcpyDeviceToHost);



	//make cumulative sum of norm_histogram

/*
	for (int i = 0; i < LIMIT + 1; ++i) {
		printf("%lf ", h_norm_histogram[i]);
	}*/



	Mat gray_scale(h, w, CV_8U, gray);
	//Mat gray_scale_eq(h, w, CV_8U, h_eqhist_image);
	imshow("frame", gray_scale);
	//imshow("eq", gray_scale_eq);
	waitKey(0);


    return 0;
}
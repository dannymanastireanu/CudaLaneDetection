#include "ImageProcessing.cuh"



__global__ void convertToGrayscale(unsigned char *gray, unsigned char *r, unsigned char *g, unsigned char *b, int dimension) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dimension)
		gray[index] = 1 / 3.0 * (r[index] + g[index] + b[index]);
}


__global__ void getHistrogram(unsigned int *histogram, unsigned char *image, int dimension) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dimension) {
		int color = image[index];
		atomicAdd(&histogram[color], 1);
	}
}


__global__ void getNormalizedHistogram(double *norm_histogram, unsigned int* histogram, int dimension) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < LIMIT + 1) {
		norm_histogram[index] = (double)histogram[index] / dimension;
	}
}


__global__ void histogramEqualization(unsigned char *eq_image, unsigned char* image, double *cumulative_sum, int dimension) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dimension) {
		eq_image[index] = floor(LIMIT * cumulative_sum[image[index]]);
	}
}


// Exclusive scan on CUDA.
__global__ void exclusiveScanGPU(int *d_array, int *d_result, int N, int *d_aux) {

	extern __shared__ int temp[];

	int realIndex = 2 * threadIdx.x + blockDim.x * 2 * blockIdx.x;

	int threadIndex = threadIdx.x;
	int index = 2 * threadIndex;

	int offset = 1;

	// Copy from the array to shared memory.
	temp[index] = d_array[realIndex];
	temp[index + 1] = d_array[realIndex + 1];

	// Reduce by storing the intermediate values. The last element will be 
	// the sum of n-1 elements.
	for (int d = blockDim.x; d > 0; d = d / 2) {
		__syncthreads();

		// Regulates the amount of threads operating.
		if (threadIndex < d)
		{
			// Swap the numbers
			int current = offset * (index + 1) - 1;
			int next = offset * (index + 2) - 1;
			temp[next] += temp[current];
		}

		// Increase the offset by multiple of 2.
		offset *= 2;
	}

	// Only one thread performs this.
	if (threadIndex == 0) {
		// Store the sum to the auxiliary array.
		if (d_aux) {
			d_aux[blockIdx.x] = temp[N - 1];
		}
		// Reset the last element with identity. Only the first thread will do
		// the job.
		temp[N - 1] = 0;
	}

	// Down sweep to build scan.
	for (int d = 1; d < blockDim.x * 2; d *= 2) {

		// Reduce the offset by division of 2.
		offset = offset / 2;

		__syncthreads();

		if (threadIndex < d)
		{
			int current = offset * (index + 1) - 1;
			int next = offset * (index + 2) - 1;

			// Swap
			int tempCurrent = temp[current];
			temp[current] = temp[next];
			temp[next] += tempCurrent;
		}
	}

	__syncthreads();

	d_result[realIndex] = temp[index]; // write results to device memory  
	d_result[realIndex + 1] = temp[index + 1];
}


cudaError_t bgrToGrayscale(unsigned char *gray, Mat image_rgb, unsigned int size)
{
	//	Host input vectors.
	unsigned char *red = new unsigned char[size];
	unsigned char *green = new unsigned char[size];
	unsigned char *blue = new unsigned char[size];



	// Init vectors with rgb values.
	for (int y = 0; y < image_rgb.rows; ++y) {
		for (int x = 0; x < image_rgb.cols; ++x) {
			blue[y * image_rgb.cols + x] = image_rgb.data[image_rgb.channels() * (y * image_rgb.cols + x) + 0];
			green[y * image_rgb.cols + x] = image_rgb.data[image_rgb.channels() * (y * image_rgb.cols + x) + 1];
			red[y * image_rgb.cols + x] = image_rgb.data[image_rgb.channels() * (y * image_rgb.cols + x) + 2];
		}
	}

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	//	Device input vectors.
	unsigned char *d_red;
	unsigned char *d_green;
	unsigned char *d_blue;
	unsigned char *d_gray;


	// Allocate GPU buffers.
	cudaStatus = cudaMalloc(&d_red, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_green, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc(&d_blue, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc(&d_gray, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers
	cudaStatus = cudaMemcpy(d_red, red, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_green, green, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_blue, blue, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	int no_threads = 1024;
	int no_blocks = (int)ceil((float)size / no_threads);


	convertToGrayscale << <no_blocks, no_threads >> > (d_gray, d_red, d_green, d_blue, size);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convert_to_grayscale launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(gray, d_gray, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_red);
	cudaFree(d_green);
	cudaFree(d_blue);
	cudaFree(d_gray);
	delete[] red;
	delete[] green;
	delete[] blue;

	return cudaStatus;
}

cudaError_t getHistogramN(double *norm_histogram, unsigned int *histogram, unsigned char *grayScaleImage, int size) {

	cudaError_t cudaStatus;

	unsigned int *d_histogram;
	unsigned char *d_gray_scale_image;
	double *d_norm_histogram;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_histogram, (LIMIT + 1) * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_norm_histogram, (LIMIT + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_gray_scale_image, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_gray_scale_image, grayScaleImage, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(d_histogram, 0, LIMIT + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(d_norm_histogram, 0, LIMIT + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}


	int no_threads = 1024;
	int no_blocks = (int)ceil((float)size / no_threads);

	getHistrogram << <no_blocks, no_threads >> > (d_histogram, d_gray_scale_image, size);
	getNormalizedHistogram << <1, 256 >> > (d_norm_histogram, d_histogram, size);

	cudaStatus = cudaMemcpy(histogram, d_histogram, (LIMIT + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(norm_histogram, d_norm_histogram, (LIMIT + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_gray_scale_image);
	cudaFree(d_histogram);

	return cudaStatus;
}

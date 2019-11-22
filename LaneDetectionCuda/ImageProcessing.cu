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
		if (floor(LIMIT * cumulative_sum[image[index]]) < LIMIT || floor(LIMIT * cumulative_sum[image[index]]) > 0)
			eq_image[index] = floor(LIMIT * cumulative_sum[image[index]]);
		else
			eq_image[index] = image[index];
	}
}


// Exclusive scan on CUDA.
__global__ void exclusiveScanGPU(double *d_array, double *d_result, int N, double *d_aux) {


	extern __shared__ double temp[];

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
			double tempCurrent = temp[current];
			temp[current] = temp[next];
			temp[next] += tempCurrent;
		}
	}

	__syncthreads();

	d_result[realIndex] = temp[index]; // write results to device memory  
	d_result[realIndex + 1] = temp[index + 1];
}

__global__ void sobelFilter(unsigned char * image, unsigned char * filtered_image, int height, int width) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	double dx, dy;

	if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
		dx = (-1 * image[(y - 1) * width + (x - 1)]) + (-2 * image[y * width + (x - 1)]) + (-1 * image[(y + 1) * width + (x - 1)]) +
			(image[(y - 1) * width + (x + 1)]) + (2 * image[y * width + (x + 1)]) + (image[(y + 1) * width + (x + 1)]);

		dy = (image[(y - 1) * width + (x - 1)]) + (2 * image[(y - 1) * width + x]) + (image[(y - 1) * width + (x + 1)]) +
			(-1 * image[(y + 1) * width + (x - 1)]) + (-2 * image[(y + 1) * width + x]) + (-1 * image[(y + 1) * width + (x + 1)]);

		filtered_image[y * width + x] = sqrt(dx * dx + dy * dy);
	}

}

__global__ void gaussianBlur(unsigned char *image, unsigned char *output_image, int width, int height, const int* const kernel, const int dim_kernel, int sum_of_elements){

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	float partial_sum = 0.0;

	if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
		for (int row = 0; row < dim_kernel; row++) {
			for (int col = 0; col < dim_kernel; col++) {
				int index_image_x = x + col - dim_kernel / 2;
				int index_image_y = y + row - dim_kernel / 2;
				index_image_x = min(max(index_image_x, 0), width - 1);
				index_image_y = min(max(index_image_y, 0), height - 1);

				partial_sum += kernel[row * dim_kernel + col] * image[index_image_y * width + index_image_x];
			}
		}

		output_image[y * width + x] = int((float)partial_sum / sum_of_elements);
	}
}

__global__ void binaryThreshold(unsigned char * image, unsigned char * output_image, int width, int height, int threshold) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
		if (image[y * width + x] < threshold)
			output_image[y * width + x] = 0;
		else
			output_image[y * width + x] = 255;
	}
}

__device__ int min_int(int a, int b) {
	return a <= b ? a : b;
}

__device__ int max_int(int a, int b) {
	return a >= b ? a : b;
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

cudaError_t getHistogramN(double *cumulativeSumHistogram, double *norm_histogram, unsigned int *histogram, unsigned char *grayScaleImage, int size) {

	cudaError_t cudaStatus;

	unsigned int *d_histogram;
	unsigned char *d_gray_scale_image;
	double *d_norm_histogram;
	double *d_cumulative_sum;
	double *d_aux_for_cumulative_sum;


	// Threads size
	int threads = 256;
	int N = 256; // Size of the array.
	int blocks = N / threads + ((N%threads == 0) ? 0 : 1);
	// Perform on CUDA.
	const dim3 blockSize(threads / 2, 1, 1);
	const dim3 gridSize(blocks, 1, 1);


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_aux_for_cumulative_sum, (LIMIT + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_cumulative_sum, (LIMIT + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_norm_histogram, (LIMIT + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_histogram, (LIMIT + 1) * sizeof(unsigned int));
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

	cudaStatus = cudaMemset(d_cumulative_sum, 0, LIMIT + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(d_aux_for_cumulative_sum, 0, LIMIT + 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}


	int no_threads = 1024;
	int no_blocks = (int)ceil((float)size / no_threads);

	getHistrogram << <no_blocks, no_threads >> > (d_histogram, d_gray_scale_image, size);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getHistrogram!\n", cudaStatus);
        goto Error;
    }
    
	getNormalizedHistogram << <1, 256 >> > (d_norm_histogram, d_histogram, size);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getNormalizedHistogram!\n", cudaStatus);
		goto Error;
	}


	exclusiveScanGPU << < gridSize, blockSize, blocks * threads * sizeof(double) >> > (d_norm_histogram, d_cumulative_sum, N, d_aux_for_cumulative_sum);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching exclusiveScanGPU!\n", cudaStatus);
		goto Error;
	}


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "exclusiveScanGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

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

	cudaStatus = cudaMemcpy(cumulativeSumHistogram, d_cumulative_sum, N * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(d_gray_scale_image);
	cudaFree(d_histogram);
	cudaFree(d_norm_histogram);
	cudaFree(d_cumulative_sum);
	cudaFree(d_aux_for_cumulative_sum);

	return cudaStatus;
}

cudaError_t doHistogramEqualization(unsigned char *eq_image, unsigned char *image, double *cumulative_sum, int dimension) {

	cudaError_t cudaStatus;

	unsigned char *d_eq_image;
	unsigned char *d_image;
	double *d_cumulative_sum;

	int no_thread = 1024;
	int no_block = (int)ceil((float)dimension / no_thread);


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_eq_image, dimension * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_image, dimension * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_cumulative_sum, (LIMIT + 1) * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_image, image, dimension * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_cumulative_sum, cumulative_sum, (LIMIT + 1) * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	histogramEqualization << <no_block, no_thread >> > (d_eq_image, d_image, d_cumulative_sum, dimension);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "histogramEqualization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching histogramEqualization!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(eq_image, d_eq_image, dimension * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_cumulative_sum);
	cudaFree(d_eq_image);
	cudaFree(d_image);

	return cudaStatus;
}

cudaError_t applySobelFilter(unsigned char *image, unsigned char *filtered_image, int width, int height) {

	cudaError_t cudaStatus;

	unsigned char *d_image;
	unsigned char *d_filtered_image;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_image, width * height * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_filtered_image, width * height * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	double number_of_threads = 32;

	dim3 threadsPerBlock(number_of_threads, number_of_threads, 1);
	dim3 numBlocks(ceil(width / number_of_threads), ceil(height / number_of_threads), 1);

	sobelFilter << <numBlocks, threadsPerBlock >> > (d_image, d_filtered_image, height, width);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sobelFilter launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sobelFilter!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(filtered_image, d_filtered_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_filtered_image);
	cudaFree(d_image);

	return cudaStatus;
}

cudaError_t applyGaussianFilter(unsigned char *image, unsigned char *filtered_image, int width, int height, const int dim_kernel) {
	int kernel[25] = {
		   1, 4, 6, 4, 1,
		   4, 16, 24, 16, 4,
		   6, 24, 36, 24, 6,
		   4, 16, 24, 16, 4,
		   1, 4, 6, 4, 1
	};

	cudaError_t cudaStatus;

	unsigned char *d_image;
	unsigned char *d_filtered_image;
	int *d_kernel;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_image, width * height * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_filtered_image, width * height * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_kernel, dim_kernel * dim_kernel * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_kernel, kernel, dim_kernel * dim_kernel * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	double number_of_threads = 32;

	dim3 threadsPerBlock(number_of_threads, number_of_threads, 1);
	dim3 numBlocks(ceil(width / number_of_threads), ceil(height / number_of_threads), 1);

	gaussianBlur << <numBlocks, threadsPerBlock >> > (d_image, d_filtered_image, width, height, d_kernel, dim_kernel, 256);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "gaussianBlur launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching gaussianBlur!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(filtered_image, d_filtered_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_filtered_image);
	cudaFree(d_image);
	cudaFree(d_kernel);

	return cudaStatus;
}

cudaError_t applyBinaryThreshold(unsigned char * image, unsigned char * filtered_image, int width, int height, const int threshold)
{
	cudaError_t cudaStatus;

	unsigned char *d_image;
	unsigned char *d_filtered_image;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_image, width * height * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_filtered_image, width * height * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	double number_of_threads = 32;

	dim3 threadsPerBlock(number_of_threads, number_of_threads, 1);
	dim3 numBlocks(ceil(width / number_of_threads), ceil(height / number_of_threads), 1);

	binaryThreshold << <numBlocks, threadsPerBlock >> > (d_image, d_filtered_image, width, height, threshold);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "binaryThreshold launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching binaryThreshold!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(filtered_image, d_filtered_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_filtered_image);
	cudaFree(d_image);

	return cudaStatus;
}

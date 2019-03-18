#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_ops.h>

#define TILE_WIDTH 16
#define MAX_KERNEL_SIZE 11
#define MAX_KERNEL_SIZE_SQUARED (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)

using namespace std;

//mask in constant memory
__constant__ float g_kernel_memory[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__ void convolution_gpu(float *d_image_in, float *d_image_out, int width, int height, int kernel_radius)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int loc = x + y * width;
	float accumulator = 0.f;

	if (x >= width || y >= height) return;
	
	for (int i = -kernel_radius; i <= kernel_radius; i++) {
		for (int j = -kernel_radius; j <= kernel_radius; j++) {
			if ((x + i < 0) || //left side out of bounds
				(x + i >= width) || //right side OoB
				(y + j < 0) || //top OoB
				(y + j >= height)) //bot OoB
				continue;
			accumulator += d_image_in[loc + i + j * width] * g_kernel_memory[i + kernel_radius + (j + kernel_radius) * ((kernel_radius << 1) + 1)];
		}
	}
	d_image_out[loc] = accumulator;
}

template <typename T1, typename T2>
void initialize_gpu_buffers(T1* h_image_in, T1** d_image_in, T2** d_image_out, int im_width, int im_height) {
	cudaMalloc((void **)d_image_in, im_width * im_height * sizeof(T1));
	cudaMalloc((void **)d_image_out, im_width * im_height * sizeof(T2));
	cudaMemcpy(*d_image_in, h_image_in, im_width * im_height * sizeof(T1), cudaMemcpyHostToDevice);
}

template <typename T>
void initialize_gpu_buffer(T* h_image_in, T** d_image_in, int im_width, int im_height) {
	cudaMalloc((void **)d_image_in, im_width * im_height * sizeof(T));
	cudaMemcpy(*d_image_in, h_image_in, im_width * im_height * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void create_gpu_buffer(T** d_image, int im_width, int im_height) {
	cudaMalloc((void **)d_image, im_width * im_height * sizeof(T));
}

template <typename T1, typename T2>
void free_gpu_buffers(T1* d_image_in, T2* d_image_out) {
	cudaFree(d_image_in);
	cudaFree(d_image_out);
	cudaFree(g_kernel_memory);
}

template <typename T>
void free_gpu_buffer(T* d_image) {
	cudaFree(d_image);
}

template <typename T>
void move_gpu_data_to_host(T* h_image_out, T* d_image_out, int im_width, int im_height) {
	cudaMemcpy(h_image_out, d_image_out, im_width * im_height * sizeof(T), cudaMemcpyDeviceToHost);
}

void convolve(float* d_image_in, float* d_image_out, float* h_kernel_data, int im_width, int im_height, int kernel_size) {
	
	cudaMemcpyToSymbol(g_kernel_memory, h_kernel_data, kernel_size * kernel_size * sizeof(float));

	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));

	convolution_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in, d_image_out, im_width, im_height, int(kernel_size / 2));
}


__global__ void median_filter_gpu(float *d_image_in, float *d_image_out, int width, int height, int filter_size)
{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float filterVector[MAX_KERNEL_SIZE_SQUARED] = { 0 };   //Take filter window
	if ((row == 0) || (col == 0) || (row == height - 1) || (col == width - 1))
		d_image_out[row*width + col] = 0; //Deal with boundry conditions
	else {
		for (int x = 0; x < filter_size; x++) {
			for (int y = 0; y < filter_size; y++) {
				filterVector[x*filter_size + y] = d_image_in[(row + x - 1)*width + (col + y - 1)];   // setup the filterign window.
			}
		}
		for (int i = 0; i < filter_size * filter_size; i++) {
			for (int j = i + 1; j < filter_size * filter_size; j++) {
				if (filterVector[i] > filterVector[j]) {
					//Swap the variables.
					float tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		d_image_out[row*width + col] = filterVector[(filter_size * filter_size) / 2];   //Set the output variables.
	}
}

void median_filter(float * d_image_in, float * d_image_out, int im_width, int im_height, int kernel_size) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	median_filter_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in, d_image_out, im_width, im_height, kernel_size);
}

__global__ void greater_than_gpu(float *d_image_in_a, float *d_image_in_b, bool *d_image_out, int width, int height, float offset) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;
	
	if (x >= width || y >= height) return;
	
	if (d_image_in_a[loc] > d_image_in_b[loc] + offset)
		d_image_out[loc] = true;
	else
		d_image_out[loc] = false;
}

void greater_than(float * d_image_in_a, float * d_image_in_b, bool* d_image_out, int im_width, int im_height, float offset) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	greater_than_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_a, d_image_in_b, d_image_out, im_width, im_height, offset);
}

__global__ void logical_xor_gpu(bool *d_image_in_a, bool *d_image_in_b, bool *d_image_out, int width, int height) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;

	d_image_out[loc] = d_image_in_a[loc] ^ d_image_in_b[loc];
}

__global__ void logical_or_gpu(bool *d_image_in_a, bool *d_image_in_b, bool *d_image_out, int width, int height) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;

	d_image_out[loc] = d_image_in_a[loc] | d_image_in_b[loc];
}

void logical_operation(bool * d_image_in_a, bool * d_image_in_b, bool* d_image_out, int im_width, int im_height, BooleanOperation operation) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	if (operation == BooleanOperation::XOR)
		logical_xor_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_a, d_image_in_b, d_image_out, im_width, im_height);
	else if (operation == BooleanOperation::OR)
		logical_or_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_a, d_image_in_b, d_image_out, im_width, im_height);
}

template void initialize_gpu_buffers(float * h_image_in, float** d_image_in, float** d_image_out, int im_width, int im_height);
template void initialize_gpu_buffers(float * h_image_in, float** d_image_in, bool** d_image_out, int im_width, int im_height);
template void initialize_gpu_buffers(bool* h_image_in, bool** d_image_in, bool** d_image_out, int im_width, int im_height);

template void initialize_gpu_buffer(float* h_image_in, float** d_image_in, int im_width, int im_height);
template void initialize_gpu_buffer(bool* h_image_in, bool** d_image_in, int im_width, int im_height);

template void create_gpu_buffer(float** d_image, int im_width, int im_height);
template void create_gpu_buffer(bool** d_image, int im_width, int im_height);

template void free_gpu_buffers(float* d_image_in, float* d_image_out);
template void free_gpu_buffers(float* d_image_in, bool* d_image_out);
template void free_gpu_buffers(bool* d_image_in, bool* d_image_out);

template void free_gpu_buffer(float* d_image);
template void free_gpu_buffer(bool* d_image);

template void move_gpu_data_to_host(float* h_image_out, float* d_image_out, int im_width, int im_height);
template void move_gpu_data_to_host(bool* h_image_out, bool* d_image_out, int im_width, int im_height);

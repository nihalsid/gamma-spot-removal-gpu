#include <iostream>
#include <cstdlib>
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
	gpuErrchk(cudaMalloc((void **)d_image_in, im_width * im_height * sizeof(T1)));
	gpuErrchk(cudaMalloc((void **)d_image_out, im_width * im_height * sizeof(T2)));
	gpuErrchk(cudaMemcpy(*d_image_in, h_image_in, im_width * im_height * sizeof(T1), cudaMemcpyHostToDevice));
}

template <typename T>
void initialize_gpu_buffer(T* h_image_in, T** d_image_in, int im_width, int im_height) {
	gpuErrchk(cudaMalloc((void **)d_image_in, im_width * im_height * sizeof(T)));
	gpuErrchk(cudaMemcpy(*d_image_in, h_image_in, im_width * im_height * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void create_gpu_buffer(T** d_image, int im_width, int im_height) {
	gpuErrchk(cudaMalloc((void **)d_image, im_width * im_height * sizeof(T)));
}

template <typename T1, typename T2>
void free_gpu_buffers(T1* d_image_in, T2* d_image_out) {
	gpuErrchk(cudaFree(d_image_in));
	gpuErrchk(cudaFree(d_image_out));
	gpuErrchk(cudaFree(g_kernel_memory));
}

template <typename T>
void free_gpu_buffer(T* d_image) {
	gpuErrchk(cudaFree(d_image));
}

template <typename T>
void move_gpu_data_to_host(T* h_image_out, T* d_image_out, int im_width, int im_height) {
	gpuErrchk(cudaMemcpy(h_image_out, d_image_out, im_width * im_height * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void move_host_data_to_gpu(T* d_image_out, T* h_image_in, int im_width, int im_height) {
	gpuErrchk(cudaMemcpy(d_image_out, h_image_in, im_width * im_height * sizeof(T), cudaMemcpyHostToDevice));
}

void set_to_true(bool* d_buffer, int im_width, int im_height) {
	gpuErrchk(cudaMemset(d_buffer, true, im_height * im_width * sizeof(bool)));
}

void convolve(float* d_image_in, float* d_image_out, float* h_kernel_data, int im_width, int im_height, int kernel_size) {
	gpuErrchk(cudaMemcpyToSymbol(g_kernel_memory, h_kernel_data, kernel_size * kernel_size * sizeof(float)));

	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));

	convolution_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in, d_image_out, im_width, im_height, int(kernel_size / 2));
}


__global__ void median_filter_gpu(float *d_image_in, float *d_image_out, int width, int height, int filter_size, bool* d_mask)
{
	// Set row and colum for thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= width || row >= height) return;
	if (!d_mask[row * width + col]) return;

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

void median_filter(float * d_image_in, float * d_image_out, int im_width, int im_height, int kernel_size, bool* d_mask) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	median_filter_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in, d_image_out, im_width, im_height, kernel_size, d_mask);
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

__global__ void less_than_constant_gpu(float *d_image_in, bool *d_image_out, int width, int height, float offset) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;

	if (d_image_in[loc] < offset)
		d_image_out[loc] = true;
	else
		d_image_out[loc] = false;
}

void less_than_constant(float * d_image_in, bool* d_image_out, int im_width, int im_height, float offset) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	less_than_constant_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in, d_image_out, im_width, im_height, offset);
}

__global__ void greater_than_constant_gpu(float *d_image_in, bool *d_image_out, int width, int height, float offset) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;

	if (d_image_in[loc] > offset)
		d_image_out[loc] = true;
	else
		d_image_out[loc] = false;
}

void greater_than_constant(float * d_image_in, bool* d_image_out, int im_width, int im_height, float offset) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	greater_than_constant_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in, d_image_out, im_width, im_height, offset);
}


template <typename T>
__global__ void multiply_constant_gpu(T *d_image_in, float* d_image_out, int width, int height, float constant_multiplier) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;
	d_image_out[loc] = (float)d_image_in[loc] * constant_multiplier;
}

template <typename T>
void multiply_constant(T *d_image_in, float* d_image_out, int im_width, int im_height, float constant_multiplier) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));

	multiply_constant_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in, d_image_out, im_width, im_height, constant_multiplier);
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

__global__ void logical_and_gpu(bool *d_image_in_a, bool *d_image_in_b, bool *d_image_out, int width, int height) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;

	d_image_out[loc] = d_image_in_a[loc] && d_image_in_b[loc];
}

void logical_operation(bool * d_image_in_a, bool * d_image_in_b, bool* d_image_out, int im_width, int im_height, BooleanOperation operation) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	if (operation == BooleanOperation::XOR)
		logical_xor_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_a, d_image_in_b, d_image_out, im_width, im_height);
	else if (operation == BooleanOperation::OR)
		logical_or_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_a, d_image_in_b, d_image_out, im_width, im_height);
	else if (operation == BooleanOperation::AND)
		logical_and_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_a, d_image_in_b, d_image_out, im_width, im_height);
}

__global__ void clear_borders_gpu(bool* d_image_in_out, int width, int height, int border_size) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;
	// -1 -> x can be width-1, -2 -> x can be width-2, width-1 
	if (!((x < border_size || x >= (width - border_size)) && (y < border_size || y >= (height - border_size)))) return;
	d_image_in_out[loc] = false;
}

void clear_borders(bool *d_image_in_out, int im_width, int im_height, int border_size) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	//numBlocks should probably be a multiple of warp size here for proper coalesce..
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	clear_borders_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_out, im_width, im_height, border_size);
}

template <typename T>
void clone_buffer(T* d_image_in, T* d_image_out, int im_width, int im_height) {
	cudaMemcpy(d_image_out, d_image_in, sizeof(T) * im_width * im_height, cudaMemcpyDeviceToDevice);
}

__global__ void masked_assign_gpu(float* d_image_in_out_a, float* d_image_in_b, int width, int height, bool* mask) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int loc = x + y * width;

	if (x >= width || y >= height) return;
	
	if (mask[loc]) {
		d_image_in_out_a[loc] = d_image_in_b[loc];
	}

}

void masked_assign(float* d_image_in_out_a, float* d_image_in_b, int im_width, int im_height, bool* mask) {
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(ceil((float)im_width / threadsPerBlock.x), ceil((float)im_height / threadsPerBlock.y));
	masked_assign_gpu <<< numBlocks, threadsPerBlock >>> (d_image_in_out_a, d_image_in_b, im_width, im_height, mask);
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

template void move_host_data_to_gpu(float* d_image_out, float* h_image_in, int im_width, int im_height);

template void multiply_constant(float *d_image_in, float* d_image_out, int im_width, int im_height, float constant_multiplier);
template void multiply_constant(bool *d_image_in, float* d_image_out, int im_width, int im_height, float constant_multiplier);

template void clone_buffer(float* d_image_in, float* d_image_out, int im_width, int im_height);

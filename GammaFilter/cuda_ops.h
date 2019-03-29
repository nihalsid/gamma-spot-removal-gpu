#pragma once
#include <cuda_runtime.h>

enum BooleanOperation {
	OR,
	XOR,
	AND
};


template <typename T1, typename T2>
void initialize_gpu_buffers(T1 * h_image_in, T1** d_image_in, T2** d_image_out, int im_width, int im_height);

template <typename T>
void initialize_gpu_buffer(T* h_image_in, T** d_image_in, int im_width, int im_height);

template <typename T>
void create_gpu_buffer(T** d_image, int im_width, int im_height);

template <typename T1, typename T2>
void free_gpu_buffers(T1* d_image_in, T2* d_image_out);

template <typename T>
void free_gpu_buffer(T* d_image);

template <typename T>
void move_gpu_data_to_host(T* h_image_out, T* d_image_out, int im_width, int im_height);

template <typename T>
void move_host_data_to_gpu(T* d_image_out, T* h_image_in, int im_width, int im_height);


template <typename T>
void multiply_constant(T *d_image_in, float* d_image_out, int im_width, int im_height, float constant_multiplier);

template <typename T>
void clone_buffer(T* d_image_in, T* d_image_out, int im_width, int im_height);

void convolve(float* d_image_in, float* d_image_out, float* h_kernel_data, int im_width, int im_height, int kernel_size);
void median_filter(float * d_image_in, float * d_image_out, int im_width, int im_height, int kernel_size, bool* d_mask);
void greater_than(float * d_image_in_a, float * d_image_in_b, bool * d_image_out, int im_width, int im_height, float offset);
void greater_than_constant(float * d_image_in, bool* d_image_out, int im_width, int im_height, float offset);
void less_than_constant(float * d_image_in, bool * d_image_out, int im_width, int im_height, float offset);
void logical_operation(bool * d_image_in_a, bool * d_image_in_b, bool* d_image_out, int im_width, int im_height, BooleanOperation operation);
void clear_borders(bool *d_image_in_out, int im_width, int im_height, int border_size);
void set_to_true(bool* d_buffer, int im_width, int im_height);
void masked_assign(float* d_image_in_out_a, float* d_image_in_b, int im_width, int im_height, bool* mask);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


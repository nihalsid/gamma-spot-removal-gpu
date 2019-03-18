#pragma once

enum BooleanOperation {
	OR,
	XOR
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

void convolve(float* d_image_in, float* d_image_out, float* h_kernel_data, int im_width, int im_height, int kernel_size);
void median_filter(float * d_image_in, float * d_image_out, int im_width, int im_height, int kernel_size);
void greater_than(float * d_image_in_a, float * d_image_in_b, bool * d_image_out, int im_width, int im_height, float offset);
void logical_operation(bool * d_image_in_a, bool * d_image_in_b, bool* d_image_out, int im_width, int im_height, BooleanOperation operation);
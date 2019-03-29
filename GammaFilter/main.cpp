#include <iostream>
// CUDA runtime
#include <cuda_runtime.h>
#include <CCfits/CCfits>
#include <npp.h>

// Utilities and system includes
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "cuda_ops.h"

#define ERROR_CHECK \
	gpuErrchk(cudaPeekAtLastError()); \
	gpuErrchk(cudaDeviceSynchronize());

void dumpToFile(std::string outfilename, std::shared_ptr<float> array_out, std::vector<int> dims) {
	std::ofstream fileout;
	int num_elements = 1;
	fileout.open(outfilename);
	for (int i = 0; i < dims.size(); i++) {
		fileout << dims[i] << " ";
		num_elements *= dims[i];
	}
	fileout << "\n";
	for (int i = 0; i < num_elements; i++) {
		fileout << array_out.get()[i] << " ";
	}
	fileout << "\n";
	fileout.close();
}

cv::Mat readImage(std::string filepath)
{
	std::auto_ptr<CCfits::FITS> pInfile(new CCfits::FITS(filepath, CCfits::Read, true));
	CCfits::PHDU& image = pInfile->pHDU();
	std::valarray<unsigned long>  contents;

	// read all user-specifed, coordinate, and checksum keys in the image
	image.readAllKeys();
	image.read(contents);
	// this doesn't print the data, just header info.
	// std::cout << contents.size() << std::endl;

	cv::Mat mat(image.axis(1), image.axis(0), CV_16UC1);
	for (int i = 0; i < image.axis(1); i++) {
		for (int j = 0; j < image.axis(0); j++) {
			mat.at<unsigned short>(i, j) = contents[i * image.axis(0) + j];
		}
	}
	// std::cout << mat.at<unsigned short>(0, 0) << ", " << mat.at<unsigned short>(0, 1) << ", " << mat.at<unsigned short>(0, 2) << "\n";
	return mat;
}

# define M_PI 3.14159265358979323846
#define FLAT_ACCESS(R, C, W) ((R) * (W) + (C))

std::shared_ptr<float> createLogFilter(int N, float sigma) {
	if (N % 2 == 0) {
		N++;
	}
	int N2 = N / 2;
	std::shared_ptr<float> G(new float[N * N], std::default_delete<float[]>());
	std::shared_ptr<float> log(new float[N * N], std::default_delete<float[]>());

	float sum_G = 0.f;
	for (int i = -N2; i <= N2; i++) {
		for (int j = -N2; j <= N2; j++) {
			G.get()[FLAT_ACCESS(i + N2, j + N2, N)] = expf(-(i*i + j * j) / (2.0 * sigma * sigma));
			sum_G += G.get()[FLAT_ACCESS(i + N2, j + N2, N)];
		}
	}

	for (int i = -N2; i <= N2; i++) {
		for (int j = -N2; j <= N2; j++) {
			log.get()[FLAT_ACCESS(i + N2, j + N2, N)] = -1 * (i * i + j * j - 2 * sigma * sigma) * G.get()[FLAT_ACCESS(i + N2, j + N2, N)] / (2 * M_PI * powf(sigma, 6) * sum_G);
		}
	}

	return log;
}

float* createBoxFilter(int N) {
	float* box = (float*)malloc(sizeof(float) * N * N);
	for (int i = 0; i < N * N; i++) {
		box[i] = 1 / (N * N);
	}
	return box;
}

void test_convolution() {
	float* hostInputImageData;
	float* hostOutputImageData;
	float* deviceInputImageData = nullptr;
	float* deviceOutputImageData = nullptr;

	const int filter_size = 9;
	std::shared_ptr<float> log_filter = createLogFilter(filter_size, 0.8);
	cv::Mat input = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\test.png", cv::IMREAD_UNCHANGED);

	hostInputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			hostInputImageData[col + row * input.cols] = float(input.at<unsigned char>(row, col));
		}
	}

	std::cout << input.channels() << std::endl;

	hostOutputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));

	initialize_gpu_buffers<float, float>(hostInputImageData, &deviceInputImageData, &deviceOutputImageData, input.cols, input.rows);
	    
	convolve(deviceInputImageData, deviceOutputImageData, log_filter.get(), input.cols, input.rows, filter_size);

	move_gpu_data_to_host<float>(hostOutputImageData, deviceOutputImageData, input.cols, input.rows);


	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			std::cout << hostOutputImageData[col + row * input.cols] << " ";
		}
		std::cout << std::endl;
	}

	free_gpu_buffers<float, float>(deviceInputImageData, deviceOutputImageData);
	free(hostInputImageData);
	free(hostOutputImageData);
}

void test_median_filter() {
	float* hostInputImageData;
	float* hostOutputImageData;
	float* deviceInputImageData = nullptr;
	float* deviceOutputImageData = nullptr;
	bool* d_true_mask = nullptr;

	
	const int filter_size = 3;
	std::shared_ptr<float> log_filter = createLogFilter(filter_size, 0.8);
	cv::Mat input = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\salt_pepper.png", cv::IMREAD_UNCHANGED);

	hostInputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			hostInputImageData[col + row * input.cols] = float(input.at<unsigned char>(row, col));
		}
	}
	create_gpu_buffer<bool>(&d_true_mask, input.cols, input.rows);
	set_to_true(d_true_mask, input.cols, input.rows);

	hostOutputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));

	initialize_gpu_buffers<float, float>(hostInputImageData, &deviceInputImageData, &deviceOutputImageData, input.cols, input.rows);
	 
	median_filter(deviceInputImageData, deviceOutputImageData, input.cols, input.rows, filter_size, d_true_mask);
	   
	move_gpu_data_to_host<float>(hostOutputImageData, deviceOutputImageData, input.cols, input.rows);

	cv::Mat image_out(input.rows, input.cols, CV_8UC1);
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			image_out.at<unsigned char>(row, col) = (unsigned char) hostOutputImageData[col + row * input.cols];
		}
	}

	cv::imwrite("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\out1.png", image_out);

	free_gpu_buffers<float, float>(deviceInputImageData, deviceOutputImageData);
	free(hostInputImageData);
	free(hostOutputImageData);
}

void test_npp_dilation() {
	float* hostInputImageData;
	float* hostOutputImageData;
	float* deviceInputImageData = nullptr;
	float* deviceOutputImageData = nullptr;

	cv::Mat input = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\bin.png", cv::IMREAD_UNCHANGED);

	hostInputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			hostInputImageData[col + row * input.cols] = float(input.at<unsigned char>(row, col));
		}
	}

	hostOutputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));

	initialize_gpu_buffers<float, float>(hostInputImageData, &deviceInputImageData, &deviceOutputImageData, input.cols, input.rows);
	
	std::cout << nppiDilate3x3_32f_C1R(deviceInputImageData, input.cols * sizeof(int), deviceOutputImageData, input.cols * sizeof(int), { input.cols, input.rows });
	
	move_gpu_data_to_host<float>(hostOutputImageData, deviceOutputImageData, input.cols, input.rows);

	cv::Mat image_out(input.rows, input.cols, CV_8UC1);
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			image_out.at<unsigned char>(row, col) = (unsigned char)(hostOutputImageData[col + row * input.cols] + 0.5f);
		//	std::cout << row << "," << col << ": " << hostOutputImageData[col + row * input.cols] << std::endl;
		}
	}

	cv::imwrite("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\out1.png", image_out);

	free_gpu_buffers<float, float>(deviceInputImageData, deviceOutputImageData);
	free(hostInputImageData);
	free(hostOutputImageData);
}

void test_greater_than() {
	float* hostInputAImageData;
	float* hostInputBImageData;

	bool* hostOutputImageData;
	float* deviceInputAImageData = nullptr;
	float* deviceInputBImageData = nullptr;
	bool* deviceOutputImageData = nullptr;


	cv::Mat input_a = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\salt_pepper.png", cv::IMREAD_UNCHANGED);

	hostInputAImageData = (float *)malloc(input_a.rows * input_a.cols * sizeof(float));
	for (int row = 0; row < input_a.rows; row++) {
		for (int col = 0; col < input_a.cols; col++) {
			hostInputAImageData[col + row * input_a.cols] = float(input_a.at<unsigned char>(row, col));
		}
	}

	cv::Mat input_b = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\mask.png", cv::IMREAD_UNCHANGED);

	hostInputBImageData = (float *)malloc(input_b.rows * input_b.cols * sizeof(float));
	for (int row = 0; row < input_b.rows; row++) {
		for (int col = 0; col < input_b.cols; col++) {
			hostInputBImageData[col + row * input_b.cols] = float(input_b.at<unsigned char>(row, col));
		}
	}

	hostOutputImageData = (bool *)malloc(input_a.rows * input_a.cols * sizeof(bool));

	initialize_gpu_buffers<float, bool>(hostInputAImageData, &deviceInputAImageData, &deviceOutputImageData, input_a.cols, input_a.rows);
	initialize_gpu_buffer<float>(hostInputBImageData, &deviceInputBImageData, input_b.cols, input_b.rows);
	
	greater_than(deviceInputAImageData, deviceInputBImageData, deviceOutputImageData, input_a.cols, input_b.rows, 10);
	
	move_gpu_data_to_host<bool>(hostOutputImageData, deviceOutputImageData, input_a.cols, input_a.rows);

	cv::Mat image_out(input_a.rows, input_a.cols, CV_8UC1);
	for (int row = 0; row < input_a.rows; row++) {
		for (int col = 0; col < input_a.cols; col++) {
			image_out.at<unsigned char>(row, col) = (unsigned char)(hostOutputImageData[col + row * input_a.cols] + 0.5f);
			//std::cout << row << "," << col << ": " << hostOutputImageData[col + row * input_a.cols] << std::endl;
		}
	}

	cv::imwrite("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\out1.png", image_out);

	free_gpu_buffers<float, bool>(deviceInputAImageData, deviceOutputImageData);
	free_gpu_buffer<float>(deviceInputBImageData);
	free(hostInputAImageData);
	free(hostInputBImageData);
	free(hostOutputImageData);
}

void test_logical_or() {
	bool* hostInputAImageData;
	bool* hostInputBImageData;

	bool* hostOutputImageData;
	bool* deviceInputAImageData = nullptr;
	bool* deviceInputBImageData = nullptr;
	bool* deviceOutputImageData = nullptr;


	cv::Mat input_a = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\mask_a.png", cv::IMREAD_UNCHANGED);

	hostInputAImageData = (bool*) malloc(input_a.rows * input_a.cols * sizeof(bool));
	for (int row = 0; row < input_a.rows; row++) {
		for (int col = 0; col < input_a.cols; col++) {
			hostInputAImageData[col + row * input_a.cols] = bool(input_a.at<unsigned char>(row, col));
		}
	}

	cv::Mat input_b = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\mask_b.png", cv::IMREAD_UNCHANGED);

	hostInputBImageData = (bool*) malloc(input_b.rows * input_b.cols * sizeof(bool));
	for (int row = 0; row < input_b.rows; row++) {
		for (int col = 0; col < input_b.cols; col++) {
			hostInputBImageData[col + row * input_b.cols] = bool(input_b.at<unsigned char>(row, col));
		}
	}

	hostOutputImageData = (bool*) malloc(input_a.rows * input_a.cols * sizeof(bool));

	initialize_gpu_buffers<bool, bool>(hostInputAImageData, &deviceInputAImageData, &deviceOutputImageData, input_a.cols, input_a.rows);
	initialize_gpu_buffer<bool>(hostInputBImageData, &deviceInputBImageData, input_b.cols, input_b.rows);

	logical_operation(deviceInputAImageData, deviceInputBImageData, deviceOutputImageData, input_a.cols, input_b.rows, BooleanOperation::OR);

	move_gpu_data_to_host<bool>(hostOutputImageData, deviceOutputImageData, input_a.cols, input_a.rows);

	cv::Mat image_out(input_a.rows, input_a.cols, CV_8UC1);
	for (int row = 0; row < input_a.rows; row++) {
		for (int col = 0; col < input_a.cols; col++) {
			image_out.at<unsigned char>(row, col) = (unsigned char)(hostOutputImageData[col + row * input_a.cols] == 1 ? 255 : 0);
			//std::cout << row << "," << col << ": " << hostOutputImageData[col + row * input_a.cols] << std::endl;
		}
	}

	cv::imwrite("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\out1.png", image_out);

	free_gpu_buffers<bool, bool>(deviceInputAImageData, deviceOutputImageData);
	free_gpu_buffer<bool>(deviceInputBImageData);
	free(hostInputAImageData);
	free(hostInputBImageData);
	free(hostOutputImageData);
}

int gam_rem_adp_log() {

	// create log filter (cpu loop)
	const int log_filter_size = 9;
	const int xmin = 200;
	const int xmax = 1700;
	const int ymin = 45;
	const int ymax = 2500;
	const float thres3 = 50;
	const float thres5 = 100;
	const float thres7 = 200;

	float* d_log_filter = nullptr;
	float* d_box_filter_normalized = nullptr;
	float* d_image_in = nullptr;
	float* d_image_log = nullptr;
	float* d_image_log_m3 = nullptr;
	float* d_image_buffer_0 = nullptr;
	float* d_image_buffer_1 = nullptr;
	float* d_image_adp = nullptr;

	bool* d_image_thres3 = nullptr;
	bool* d_image_thres5 = nullptr;
	bool* d_image_thres7 = nullptr;
	bool* d_image_single7 = nullptr;
	bool* d_bool_buffer = nullptr;
	bool* d_true_mask = nullptr;

	float* h_image_in = nullptr;
	float* h_image_out = nullptr;
	float* h_single_7 = createBoxFilter(3);

	std::shared_ptr<float> log_filter = createLogFilter(log_filter_size, 0.8);

	initialize_gpu_buffer<float>(log_filter.get(), &d_log_filter, log_filter_size, log_filter_size);

	cv::Mat full_image = readImage("C:\\Users\\Yawar\\Documents\\FRM-II\\data\\ob\\00155908.fits");
	std::cout << full_image.rows << ", " << full_image.cols << std::endl;

	cv::Mat image(ymax - ymin, xmax - xmin, CV_16UC1);
	full_image(cv::Rect(xmin, ymin, image.cols, image.rows)).copyTo(image);

	std::cout << image.rows << ", " << image.cols << std::endl;
	std::cout << image.at<unsigned short>(0, 0) << ", " << image.at<unsigned short>(0, 1) << ", " << image.at<unsigned short>(0, 2) << "\n";

	h_image_in = (float *)malloc(image.rows * image.cols * sizeof(float));
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			h_image_in[col + row * image.cols] = float(image.at<unsigned short>(row, col));
		}
	}

	h_image_out = (float *)malloc(image.rows * image.cols * sizeof(float));

	initialize_gpu_buffers<float, float>(h_image_in, &d_image_in, &d_image_log, image.cols, image.rows);
	create_gpu_buffer<float>(&d_image_log_m3, image.cols, image.rows);
	create_gpu_buffer<float>(&d_image_buffer_0, image.cols, image.rows);
	create_gpu_buffer<float>(&d_image_buffer_1, image.cols, image.rows);
	create_gpu_buffer<float>(&d_image_adp, image.cols, image.rows);
	create_gpu_buffer<float>(&d_box_filter_normalized, 3, 3);
	create_gpu_buffer<bool>(&d_image_thres3, image.cols, image.rows);
	create_gpu_buffer<bool>(&d_image_thres5, image.cols, image.rows);
	create_gpu_buffer<bool>(&d_image_thres7, image.cols, image.rows);
	create_gpu_buffer<bool>(&d_image_single7, image.cols, image.rows);
	create_gpu_buffer<bool>(&d_bool_buffer, image.cols, image.rows);
	create_gpu_buffer<bool>(&d_true_mask, image.cols, image.rows);

	set_to_true(d_true_mask, image.cols, image.rows);
	move_host_data_to_gpu(d_box_filter_normalized, h_single_7, 3, 3);

	ERROR_CHECK

		convolve(d_image_in, d_image_log, log_filter.get(), image.cols, image.rows, log_filter_size);
	ERROR_CHECK
		median_filter(d_image_log, d_image_log_m3, image.cols, image.rows, 3, d_true_mask);
	ERROR_CHECK
		greater_than(d_image_log, d_image_log_m3, d_image_thres3, image.cols, image.rows, thres3);
	greater_than(d_image_log, d_image_log_m3, d_image_thres5, image.cols, image.rows, thres5);
	greater_than(d_image_log, d_image_log_m3, d_image_thres7, image.cols, image.rows, thres7);
	ERROR_CHECK

		// TODO: put a condition for non zero img_thres7

		multiply_constant(d_image_thres7, d_image_buffer_0, image.cols, image.rows, 255.f);
	ERROR_CHECK
		convolve(d_image_buffer_0, d_image_buffer_0, d_box_filter_normalized, image.cols, image.rows, 3);
	ERROR_CHECK
		less_than_constant(d_image_buffer_0, d_image_single7, image.cols, image.rows, 30.f);
	ERROR_CHECK
		logical_operation(d_image_single7, d_image_thres7, d_image_single7, image.cols, image.rows, BooleanOperation::AND);
	ERROR_CHECK
		logical_operation(d_image_thres7, d_image_single7, d_image_thres7, image.cols, image.rows, BooleanOperation::XOR);
	ERROR_CHECK
		multiply_constant(d_image_thres7, d_image_buffer_0, image.cols, image.rows, 255.f);
	ERROR_CHECK
		nppiDilate3x3_32f_C1R(d_image_buffer_0, image.cols * sizeof(int), d_image_buffer_1, image.cols * sizeof(int), { image.cols, image.rows });
	ERROR_CHECK
		greater_than_constant(d_image_buffer_1, d_bool_buffer, image.cols, image.rows, 0.);
	ERROR_CHECK
		logical_operation(d_bool_buffer, d_image_single7, d_image_thres7, image.cols, image.rows, BooleanOperation::OR);
	ERROR_CHECK

		// end of TODO:if

		logical_operation(d_image_thres5, d_image_thres7, d_bool_buffer, image.cols, image.rows, BooleanOperation::OR);
	ERROR_CHECK
		logical_operation(d_bool_buffer, d_image_thres7, d_image_thres5, image.cols, image.rows, BooleanOperation::XOR);
	ERROR_CHECK
		logical_operation(d_image_thres3, d_image_thres5, d_image_thres3, image.cols, image.rows, BooleanOperation::XOR);
	ERROR_CHECK

		clear_borders(d_image_thres7, image.cols, image.rows, 3);
	clear_borders(d_image_thres5, image.cols, image.rows, 2);
	ERROR_CHECK

		clone_buffer(d_image_in, d_image_adp, image.cols, image.rows);
	median_filter(d_image_in, d_image_buffer_0, image.cols, image.rows, 3, d_true_mask);
	ERROR_CHECK

		masked_assign(d_image_adp, d_image_buffer_0, image.cols, image.rows, d_image_thres3);
	ERROR_CHECK

		median_filter(d_image_in, d_image_adp, image.cols, image.rows, 3, d_image_thres5);
	ERROR_CHECK
		median_filter(d_image_in, d_image_adp, image.cols, image.rows, 4, d_image_thres7);
	ERROR_CHECK

		move_gpu_data_to_host<float>(h_image_out, d_image_adp, image.cols, image.rows);

	cv::Mat image_out(image.rows, image.cols, CV_16UC1);
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			image_out.at<unsigned short>(row, col) = h_image_out[col + row * image.cols];
		}
	}

	cv::imwrite("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\out1.png", image_out);

	free(h_image_in);
	free(h_image_out);

	free_gpu_buffers<float, float>(d_image_in, d_image_log);
	free_gpu_buffer<float>(d_image_log_m3);
	free_gpu_buffer<float>(d_image_buffer_0);
	free_gpu_buffer<float>(d_image_buffer_1);
	free_gpu_buffer<float>(d_image_adp);
	free_gpu_buffer<float>(d_box_filter_normalized);
	free_gpu_buffer<bool>(d_image_thres3);
	free_gpu_buffer<bool>(d_image_thres5);
	free_gpu_buffer<bool>(d_image_thres7);
	free_gpu_buffer<bool>(d_image_single7);
	free_gpu_buffer<bool>(d_bool_buffer);
	free_gpu_buffer<bool>(d_true_mask);

	return 0;
}


int main() {
	gam_rem_adp_log();
}
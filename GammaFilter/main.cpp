#include <iostream>
// CUDA runtime
#include <cuda_runtime.h>
#include <CCfits/CCfits>
#include <npp.h>

// Utilities and system includes
#include <memory>
#include <opencv2/opencv.hpp>
#include "cuda_ops.h"

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

	const int filter_size = 3;
	std::shared_ptr<float> log_filter = createLogFilter(filter_size, 0.8);
	cv::Mat input = cv::imread("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\salt_pepper.png", cv::IMREAD_UNCHANGED);

	hostInputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));
	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {
			hostInputImageData[col + row * input.cols] = float(input.at<unsigned char>(row, col));
		}
	}

	hostOutputImageData = (float *)malloc(input.rows * input.cols * sizeof(float));

	initialize_gpu_buffers<float, float>(hostInputImageData, &deviceInputImageData, &deviceOutputImageData, input.cols, input.rows);
	 
	median_filter(deviceInputImageData, deviceOutputImageData, input.cols, input.rows, filter_size);
	   
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
	const int filter_size = 9;
	std::shared_ptr<float> log_filter = createLogFilter(filter_size, 0.8);
	float* d_log_filter = nullptr;
	initialize_gpu_buffer<float>(log_filter.get(), &d_log_filter, filter_size, filter_size);
	
	//dumpToFile("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\logfilter.txt", log_filter, {9, 9});
	cv::Mat image = readImage("C:\\Users\\Yawar\\Documents\\FRM-II\\data\\ob\\00155908.fits");
	std::cout << image.rows << ", " << image.cols << std::endl;
	
	int xmin = 200;
	int xmax = 1700;
	int ymin = 45;
	int ymax = 2500;


	cv::Mat sub_image(ymax - ymin, xmax - xmin, CV_16UC1);
	image(cv::Rect(xmin, ymin, sub_image.cols, sub_image.rows)).copyTo(sub_image);

	std::cout << sub_image.rows << ", " << sub_image.cols << std::endl;
	std::cout << sub_image.at<unsigned short>(0, 0) << ", " << sub_image.at<unsigned short>(0, 1) << ", " << sub_image.at<unsigned short>(0, 2) << "\n";
	
	float* h_image_in = (float *)malloc(sub_image.rows * sub_image.cols * sizeof(float));
	for (int row = 0; row < sub_image.rows; row++) {
		for (int col = 0; col < sub_image.cols; col++) {
			h_image_in[col + row * sub_image.cols] = float(sub_image.at<unsigned short>(row, col));
		}
	}
	
	float* h_image_out = (float *)malloc(sub_image.rows * sub_image.cols * sizeof(float));
	float* d_image_in = nullptr, *d_image_out = nullptr;
	 
	initialize_gpu_buffers<float, float>(h_image_in, &d_image_in, &d_image_out, sub_image.cols, sub_image.rows);
	convolve(d_image_in, d_image_out, log_filter.get(), sub_image.cols, sub_image.rows, filter_size);
	median_filter(d_image_out, deviceOutputImageData, input.cols, input.rows, filter_size);
	move_gpu_data_to_host<float>(h_image_out, d_image_out, sub_image.cols, sub_image.rows);

	cv::Mat image_out(sub_image.rows, sub_image.cols, CV_16UC1);
	for (int row = 0; row < sub_image.rows; row++) {
		for (int col = 0; col < sub_image.cols; col++) {
			image_out.at<unsigned short>(row, col) = h_image_out[col + row * sub_image.cols];
		}
	}

	cv::imwrite("C:\\Users\\Yawar\\Documents\\FRM-II\\test\\out1.png", image_out);
	
	// fftconvolve 
	// med filter
	// threshold
	// threshold
	// threshold
	// convolve2d
	// logical xor
	
	// logical or, binary dilation
	// logical xor
	// logical xor
	// median filter
	// median filter

	free_gpu_buffers<float, float>(d_image_in, d_image_out);
	free(h_image_in);
	free(h_image_out);
	return 0;
}

int main() {
	test_logical_or();
}
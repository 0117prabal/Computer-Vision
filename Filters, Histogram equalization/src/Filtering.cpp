#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace std;
using namespace cv;

class Filtering{

private:

	Mat grayImage;

public:

	Filtering(string path){

		grayImage = imread(path, 0);

		if(!grayImage.data){

			cout<<"Unable to read the image"<<endl;

		}

		imshow("Original", grayImage);
		waitKey(0);

	}

	void run(){

		usingGaussianBlur();
		usingFilter2D();
		usingSepFilter2D();
		destroyAllWindows();
	}

	void usingGaussianBlur(){
		Mat result;
		GaussianBlur(grayImage, result, Size(0,0), 2*sqrt(2));
		imshow("Using gaussian blur", result);
		waitKey(0);
	}

	void usingFilter2D(){
		Mat result;
		Mat kernel = calculateGaussianKernel(2*sqrt(2));
		filter2D(grayImage, result, -1, kernel);
		imshow("Using filter2D", result);
		waitKey(0);
	}

	void usingSepFilter2D(){

		Mat result;
		Mat kernelX = getGaussianFilterX(2*sqrt(2));
		Mat kernelY = getGaussianFilterY(2*sqrt(2));
		sepFilter2D(grayImage, result, -1, kernelX, kernelY);
		imshow("Using sepFilter2D", result);
		waitKey(0);
	}

	Mat getGaussianFilterX(float sigma){

		int size = (int)(2.f*(((sigma - 0.8f) / 0.3f)+1.f) + 1.f);
		int radius = size/2;
		double totalSum = 0.0;

		Mat result(size, 1, CV_32F);

		for(int i = -radius ; i < radius ; i++){

			result.at<float>(i + radius,0) = (1.f / (2*M_PI*pow(sigma,2))) * exp( (-(pow(i,2)) / (2*pow(sigma,2))));
			totalSum += result.at<float>(i+radius,0);	
		}

		for(int i = 0 ; i < size ; i++){

			result.at<float>(i,0) /= totalSum;
		}

		return result;
	}

	Mat getGaussianFilterY(float sigma){

		int size = (int)(2.f*(((sigma - 0.8f) / 0.3f)+1.f) + 1.f);
		int radius = size/2;
		double totalSum = 0.0;

		Mat result(size, 1, CV_32F);

		for(int i = -radius ; i < radius ; i++){

			result.at<float>(i+radius,0) = (1.f / (2*M_PI*pow(sigma,2))) * exp( (-(pow(i,2)) / (2*pow(sigma,2))));
			totalSum += result.at<float>(i+radius,0);	
		}

		for(int i = 0 ; i < size ; i++){
			
			result.at<float>(i,0) /= totalSum;
		}

		return result;

	}

	Mat calculateGaussianKernel(float sigma){
		
		int size = (int)(2.f*(((sigma - 0.8f) / 0.3f)+1.f) + 1.f);

		int radius = size/2;
		double totalSum = 0.0;

		Mat kernel(size, size, CV_32F);

		for(int i = -radius ; i < radius ; i++){
			for(int j = -radius ; j < radius ; j++){

				kernel.at<float>(i+radius,j+radius) = (1.f / (2*M_PI*pow(sigma,2))) * exp( (-(pow(i,2) + pow(j,2))) / (2*pow(sigma,2)));
				totalSum += kernel.at<float>(i+radius,j+radius);  

			}
		}

		for(int i = 0 ; i < size ; i++){
			for(int j = 0 ; j < size ; j++){

				kernel.at<float>(i,j) /= (totalSum); 

			}
		}

		return kernel;
	}

	~Filtering(){
		//nothing to do here
	}
};
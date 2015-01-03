#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class HistogramEqualization{

private:

	Mat grayImage;

public:

	HistogramEqualization(string path){

		grayImage = imread(path, 0);

		if(!grayImage.data){
			cout<<"Unable to read the image"<<endl;
		}

		imshow("Original", grayImage);
		waitKey(0);
	}

	void run(){

		Mat result;
		equalizeHist(grayImage, result);
		imshow("Using built-in function", result);
		waitKey(0);

		showWithManual();
		destroyAllWindows();
	}

	void showWithManual(){

		Mat temp;
		
		grayImage.convertTo(temp, CV_32F);

		int histogram[255];
		int cumultativeHistogram[255];

		for(int i = 0; i < 256 ; i++){

			histogram[i] = 0;
			cumultativeHistogram[i] = 0;

		}

		for(int i = 0 ; i < grayImage.rows ; i++){
			for(int j = 0 ; j < grayImage.cols ; j++){

				histogram[(int)temp.at<float>(i,j)]++;

			}
		}

		cumultativeHistogram[0] = histogram[0];

		for(int i = 1 ; i < 256 ; i++){

				cumultativeHistogram[i] = histogram[i] + cumultativeHistogram[i-1];

		}

		Mat result(grayImage.rows, grayImage.cols, CV_32F);

		for(int i = 0 ; i < result.rows ; i++){
			for(int j = 0 ; j < result.cols ; j++){

				result.at<float>(i,j) = cumultativeHistogram[(int)temp.at<float>(i,j)] * (255.f /(result.rows * result.cols)) ;

			}
		}

		result.convertTo(result, CV_8UC1);

		imshow("Without using built-in function", result);
		waitKey(0);

	}

	~HistogramEqualization(){
		//nothing to do here
	}
};
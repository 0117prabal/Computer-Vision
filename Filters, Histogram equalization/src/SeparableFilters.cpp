#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class SeparableFilters{

private:
	
	Mat grayImage;
	Mat kernel1;
	Mat kernel2;

public:

	SeparableFilters(string path){

		grayImage = imread(path, 0);

		if(!grayImage.data){
			cout<<"Unable to read the image"<<endl;
		}

		kernel1 = (Mat_<float>(3,3) << 0.0113, 0.0838, 0.0113, 0.0838, 0.6193, 0.0838, 0.0113, 0.0838, 0.0113);
		kernel2 = (Mat_<float>(3,3) << -0.8984, 0.1472, 1.1410, -1.9075, 0.1566, 2.1359, -0.8659, 0.0573, 1.0337);

		imshow("Original", grayImage);
		waitKey(0);
	}

	void run(){
		
		processKernel1();
		processKernel2();
		destroyAllWindows();

	}

	void processKernel1(){

		Mat result;
		filter2D(grayImage, result, -1, kernel1);
		imshow("Kernel 1 result", result);
		waitKey(0);

		Mat W, U, Vt;
		SVD::compute(kernel1, W, U, Vt);

		Mat kernelX = sqrt(W.at<float>(0,0)) * U.col(0);
		Mat kernelY = sqrt(W.at<float>(0,0)) * Vt.row(0);

		Mat nresult;
		sepFilter2D(grayImage, nresult, -1, kernelX, kernelY);
		imshow("Kernel 1 separated", nresult);
		waitKey(0);

		Mat matAbsDiff;
		double minVal, maxVal;

		absdiff(result, nresult, matAbsDiff);
		minMaxLoc(matAbsDiff, &minVal, &maxVal);
		cout<<"maxVal: "<<maxVal<<endl;
	}

	void processKernel2(){

		Mat result;
		filter2D(grayImage, result, -1, kernel2);
		imshow("Kernel 2 result", result);
		waitKey(0);

		Mat W, U, Vt;
		SVD::compute(kernel2, W, U, Vt);

		Mat kernelX = sqrt(W.at<float>(0,0)) * U.col(0);
		Mat kernelY = sqrt(W.at<float>(0,0)) * Vt.row(0);

		Mat nresult;
		sepFilter2D(grayImage, nresult, -1, kernelX, kernelY);
		imshow("Kernel 2 separated", nresult);
		waitKey(0);		

		Mat matAbsDiff;
		double minVal, maxVal;

		absdiff(result, nresult, matAbsDiff);
		minMaxLoc(matAbsDiff, &minVal, &maxVal);
		cout<<"maxVal: "<<maxVal<<endl;

	}

	~SeparableFilters(){
		//nothing to do here
	}

};
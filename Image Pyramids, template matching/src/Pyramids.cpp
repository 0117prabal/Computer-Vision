#ifndef _PYRAMID
#define _PYRAMID

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Pyramids{

private:

	Mat image;
	vector<Mat> gaussianPyramid;
	vector<Mat> laplacianPyramid;
	int level;

public:

	Pyramids(string path, int level){

		image = imread(path, 0);

		if(!image.data){
			cout<<"Unable to read the image"<<endl;
		}

		this->level = level;
		run();
	}

	Pyramids(Mat image, int level){

		assert(image.data);

		this->level = level;
		this->image = image;
		run();
	}

	void run(){

		usingBuiltInFunction();
		usingManualFunction();

	}

	void usingBuiltInFunction(){

		gaussianPyramid.push_back(image);

		for(int i = 0 ; i < level ; i++){
			Mat temp, temp1;
			pyrDown(gaussianPyramid[i], temp);
			pyrUp(temp, temp1);
			gaussianPyramid.push_back(temp);
			laplacianPyramid.push_back(gaussianPyramid[i] - temp1);
		}
	}

	static void showPyramid(vector<Mat> images){

		for(int i = 0 ; i < images.size() ; i++){
			imshow((i+1)+"", images[i]);
			waitKey(0);
		}
	}

	vector<Mat> getGaussianPyramid(){
		
		return gaussianPyramid;
	}

	vector<Mat> getLaplacianPyramid(){
		
		return laplacianPyramid;
	}

	void usingManualFunction(){

		vector<Mat> gaussianPyramidManual;
		vector<Mat> laplacianPyramidManual;

		gaussianPyramidManual.push_back(image);

		for(int i = 0 ; i < level ; i++){
			Mat temp, temp1;
			GaussianBlur(gaussianPyramidManual[i], temp, Size(0,0), 2*sqrt(2));
			resize(temp, temp, Size(), 0.5, 0.5);
			pyrUp(temp, temp1);
			gaussianPyramidManual.push_back(temp);
			laplacianPyramidManual.push_back(gaussianPyramidManual[i] - temp1);
		}

	}

	~Pyramids(){
		//nothing to do here
	}

};

#endif
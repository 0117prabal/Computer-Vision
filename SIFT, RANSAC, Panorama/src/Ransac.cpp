#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

#define IMAGE1 1
#define IMAGE2 2

class Ransac{

private:

	Mat image1;
	Mat image2;

	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;

	Mat descs1;
	Mat descs2;

public:

	Ransac(){

		image1 = imread("images/image1.png", IMREAD_COLOR);
		image2 = imread("images/image2.png", IMREAD_COLOR);
		imshow("Image 1", image1);
		waitKey(0);
		imshow("Image 2", image2);
		waitKey(0);

	}

	void run(){

		applySIFT(IMAGE1);
		applySIFT(IMAGE2);

		showKeyPoints(IMAGE1);
		showKeyPoints(IMAGE2);
	}

	void applySIFT(int type){

		assert(type == IMAGE1 || type == IMAGE2);

		SIFT sift;

		if(type == IMAGE1){
			
			sift(image1, noArray(), keypoints1, descs1);

		}
		else{

			sift(image2, noArray(), keypoints2, descs2);
		}

	}

	void showKeyPoints(int type){

		assert(type == IMAGE1 || type == IMAGE2);

		Mat output;

		if(type == IMAGE1){

  			cv::drawKeypoints(image1, keypoints1, output);
			imshow("Keypoints for Image-1", output);

		}
		else{

			cv::drawKeypoints(image2, keypoints2, output);
			imshow("Keypoints for Image-2", output);

		}

		waitKey(0);

	}

	~Ransac(){

		//nothing to do here
	}

};
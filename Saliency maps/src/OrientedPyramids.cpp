#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <cmath>

#include <iostream>

using namespace std;
using namespace cv;

/*
	This class generates the oriented pyramids
*/

class OrientedPyramids{

private:

	vector <Mat> pyramidDeg0;
	vector <Mat> pyramidDeg45;
	vector <Mat> pyramidDeg90;
	vector <Mat> pyramidDeg135;

public:

	OrientedPyramids(){

	}

	void build(vector <Mat> lap){

		this->pyramidDeg0   = buildPyramid(lap, 0.f * (M_PI / 180.f));
		this->pyramidDeg45  = buildPyramid(lap, 45.f* (M_PI / 180.f));
		this->pyramidDeg90  = buildPyramid(lap, 90.f* (M_PI / 180.f));
		this->pyramidDeg135 = buildPyramid(lap, 135.f* (M_PI / 180.f));

		//showImages(pyramidDeg0);
		//showImages(pyramidDeg45);
		//showImages(pyramidDeg90);
		//showImages(pyramidDeg135);

	}

	vector <Mat> buildPyramid(vector <Mat> lap, float orientation){

		vector <Mat> result;

		for(int i = 0 ; i < lap.size() ; i++){

			Mat kernel = getGaborKernel(Size(lap[i].rows/2,lap[i].cols/2), 0.9, orientation, 5.0, 3.5);
			Mat temp;
			filter2D(lap[i], temp, -1, kernel);
			result.push_back(temp);

		}

		return result;
	}

	vector <Mat> getPyramid0(){

		return pyramidDeg0;
	}

	vector <Mat> getPyramid45(){

		return pyramidDeg45;
	}

	vector <Mat> getPyramid90(){

		return pyramidDeg90;
	}

	vector <Mat> getPyramid135(){

		return pyramidDeg135;
	}

	void showImages(vector <Mat> images){

		for(int i = 0 ; i < images.size() ; i++){

			imshow(""+i, images[i]);
			waitKey(0);

		}

	}

	void showFloatImage(Mat fImage){

		double minVal, maxVal;
		minMaxLoc(fImage, &minVal, &maxVal); //find minimum and maximum intensities
		fImage.convertTo(fImage, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
		imshow("fImage", fImage);
		waitKey(0);

	}

	static string type2str(int type) {
	  
	  string r;

	  uchar depth = type & CV_MAT_DEPTH_MASK;
	  uchar chans = 1 + (type >> CV_CN_SHIFT);

	  switch ( depth ) {
	    case CV_8U:  r = "8U"; break;
	    case CV_8S:  r = "8S"; break;
	    case CV_16U: r = "16U"; break;
	    case CV_16S: r = "16S"; break;
	    case CV_32S: r = "32S"; break;
	    case CV_32F: r = "32F"; break;
	    case CV_64F: r = "64F"; break;
	    default:     r = "User"; break;
	  }

	  r += "C";
	  r += (chans+'0');

	  return r;
	}

	~OrientedPyramids(){

	}

};
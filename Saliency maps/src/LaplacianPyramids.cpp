#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>

using namespace std;
using namespace cv;


/*
	This class generates the laplacian pyramids
*/

class LaplacianPyramids{

private:

	vector <Mat> LLpyramid;
	vector <Mat> LApyramid;
	vector <Mat> LBpyramid;

public:

	LaplacianPyramids(){

	}

	void build(vector<Mat> lpyramid, vector<Mat> apyramid, vector<Mat> bpyramid){

		LLpyramid = buildPyramid(lpyramid);
		LApyramid = buildPyramid(apyramid);
		LBpyramid = buildPyramid(bpyramid);

		//showImages(LLpyramid);
		//showImages(LApyramid);
		//showImages(LBpyramid);

	}

	void forceSize(Mat im1, Mat& im2){

		if(im1.rows == im2.rows && im1.cols == im2.cols){

			return;
		}

		resize(im2, im2, im1.size());

	}

	vector <Mat> buildPyramid(vector <Mat> pyramid){

		vector <Mat> result;

		for(int i = 0; i < pyramid.size() ; i++){

			if(i != (pyramid.size()-1)){
			
				Mat upscaled;
				pyrUp(pyramid[i+1], upscaled);
				forceSize(pyramid[i], upscaled);

				Mat temp = pyramid[i] - upscaled;

				result.push_back(temp);
			}
			else{

				result.push_back(pyramid[i]);
			}

		}

		return result;
	}

	void showImages(vector <Mat> images){

		for(int i = 0 ; i < images.size() ; i++){

			Mat temp;
			double minVal, maxVal;
			minMaxLoc(images[i], &minVal, &maxVal); //find minimum and maximum intensities
			images[i].convertTo(temp, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
			imshow(""+i, temp);
			waitKey(0);

		}

	}

	vector <Mat> getLLpyramid(){

		return LLpyramid;
	}

	vector <Mat> getLApyramid(){

		return LApyramid;
	}

	vector <Mat> getLBpyramid(){

		return LBpyramid;
	}

	~LaplacianPyramids(){

	}


};
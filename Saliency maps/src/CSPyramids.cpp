#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>

using namespace std;
using namespace cv;

/*
	This class generates the center surround and the surround center images
*/

class CSPyramids{

private:

	vector <Mat> nfPyramid;
	vector <Mat> fnPyramid;

public:

	CSPyramids(){

	}

	void build(vector <Mat> featurePyramid){
		
		float sigma = 1.f;

		vector <Mat> C = getCenter(featurePyramid, 3.f);
		vector <Mat> S = getSurround(featurePyramid, 7.f);

		this->nfPyramid = diff(C,S);
		this->fnPyramid = diff(S,C);

		//showImages(nfPyramid);
		//showImages(fnPyramid);

	}

	//function to take the difference of two vector of images
	vector <Mat> diff(vector <Mat> one, vector <Mat> two){

		assert(one.size() == two.size());
		vector <Mat> result;

		for(int i = 0 ; i < one.size() ; i++){

			Mat temp = one[i] - two[i];
			result.push_back(temp);

		}

		return result;
	}

	// generate the center images
	vector <Mat> getCenter(vector <Mat> featurePyramid, float sigma){

		vector <Mat> result;

		for(int i = 0 ; i < featurePyramid.size() ; i++){

			Mat temp;
			GaussianBlur(featurePyramid[i], temp, Size(0,0), sigma);
			result.push_back(temp);

		}

		return result;
	}


	//generate the surround images
	vector <Mat> getSurround(vector <Mat> featurePyramid, float sigma){

		vector <Mat> result;

		for(int i = 0 ; i < featurePyramid.size() ; i++){

			Mat temp;
			GaussianBlur(featurePyramid[i], temp, Size(0,0), sigma);
			result.push_back(temp);

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

	vector <Mat> getCS(){

		return nfPyramid;
	}

	vector <Mat> getSC(){

		return fnPyramid;
	}

	~CSPyramids(){

	}
};
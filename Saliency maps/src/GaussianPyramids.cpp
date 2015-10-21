#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>

using namespace std;
using namespace cv;

/*
	This class generates the gaussian pyramids
*/

class GaussianPyramids{

private:

	Mat image;
	int depth;
	vector <Mat> Lpyramid;
	vector <Mat> Apyramid;
	vector <Mat> Bpyramid;

public:

	GaussianPyramids(Mat image, int depth){

		this->image = image;
		this->depth = depth;

	}

	void build(){

		Mat newImage;
		cvtColor(image, newImage, CV_BGR2Lab);
		vector <Mat> channels(3);

		split(newImage, channels);

		Mat B = channels[2];
		Mat A = channels[1];
		Mat L = channels[0];

		/*imshow("B", B);
		waitKey(0);

		imshow("A", A);
		waitKey(0);

		imshow("L", L);
		waitKey(0);*/

		this->Bpyramid = buildPyramid(B);
		this->Apyramid = buildPyramid(A);
		this->Lpyramid = buildPyramid(L);

		//showImages(Bpyramid);
		//showImages(Apyramid);
		//showImages(Lpyramid);

	}

	vector<Mat> buildPyramid(Mat image){

		vector<Mat> result;

		for(int i = 0 ; i < depth ; i++){

			if(i == 0){
				
				Mat temp;
				pyrDown(image, temp);
				result.push_back(temp);
			}

			else{

				Mat temp;
				pyrDown(result[i-1], temp);
				result.push_back(temp);
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

	vector <Mat> getLpyramid(){

		return Lpyramid;
	}

	vector <Mat> getApyramid(){

		return Apyramid;
	}

	vector <Mat> getBpyramid(){

		return Bpyramid;
	}



	~GaussianPyramids(){

	}

};
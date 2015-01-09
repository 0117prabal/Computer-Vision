#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

class DisparityMap{

private:

	Mat image1;
	Mat image2;
	Mat disparityMap;

public:

	DisparityMap(){
		
		cout<<endl<<"Part 3"<<endl;
		image1 = imread("images/aloe1.png", 0);
		image2 = imread("images/aloe2.png", 0);

		if(!image1.data || !image2.data){

			cout<<"Unable to read the images"<<endl;

		}

		disparityMap = Mat::zeros(image1.rows, image1.cols, CV_32F);

	}

	void run(){

		int patchHeight = 10;
		int patchWidth = 7;

		//cout<<image1.rows<<endl;   370
		//cout<<image1.cols<<endl;	 427
		cout<<"Calculating disparity map"<<flush;

		for(int i =  0 ; i < image1.rows / patchHeight ; i++){
			for(int j = 0 ; j < image1.cols / patchWidth ; j++){

				Rect rect(j*patchWidth ,i*patchHeight ,patchWidth,patchHeight);
				Mat result;
				Mat patch = image1(rect).clone();
				int result_cols =  image2.cols - patch.cols + 1;
  				int result_rows = image2.rows - patch.rows + 1;
  				result.create( result_cols, result_rows, CV_32F);
				matchTemplate(image2, patch, result, CV_TM_SQDIFF_NORMED);
				normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());
				Point minLoc;
			    minMaxLoc(result, NULL, NULL, &minLoc, NULL, Mat());
			    saveToMap(patch, minLoc, i*patchHeight, j*patchWidth, patchHeight, patchWidth);

			}

			cout<<"."<<flush;
		}

		double max;
		minMaxLoc(disparityMap, NULL, &max, NULL, NULL, Mat());
		disparityMap.convertTo(disparityMap, CV_8U, 255.0/max);
		imshow("Disparity Map", disparityMap);
		waitKey(0);

	}

	void saveToMap(Mat patch, Point location, int i, int j, int patchHeight, int patchWidth){

		Mat one = getLocationMatrix(i, j, patchWidth, patchHeight);
		Mat two = getLocationMatrix(location.x, location.y, patchWidth, patchHeight);

		Mat result;
		absdiff(one, two, result);
		result.copyTo(disparityMap.rowRange(i, i+patchHeight).colRange(j, j+patchWidth));

	}

	Mat getLocationMatrix(int indexX, int indexY, int width, int height){

		Mat result(height, width, CV_32F);

		for(int i = 0 ; i < height; i++){
			for(int j = 0 ; j < width; j++){

				result.at<float>(i,j) = indexX+j;

			}
		}

		return result;
	}

	~DisparityMap(){

		//nothing to do here

	}
};
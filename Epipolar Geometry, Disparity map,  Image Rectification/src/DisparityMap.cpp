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

		for(int i =  0 ; i < image1.rows-10 ; i++){
			for(int j = 0 ; j < image1.cols-7 ; j++){

				Rect rect(j,i,patchWidth,patchHeight);
				/*rectangle(image1, Point(j*patchWidth, i*patchHeight), Point(j*patchWidth + patchWidth, i*patchHeight + patchHeight), Scalar(0, 0, 255), 1, 8, 0);
				imshow("rew",image1);
				waitKey(0);*/
				Mat result;
				Mat patch = image1(rect).clone();
				int result_cols =  image2.cols - patch.cols + 1;
  				int result_rows = image2.rows - patch.rows + 1;
  				result.create( result_cols, result_rows, CV_32F);
				matchTemplate(image2, patch, result, CV_TM_SQDIFF_NORMED);
				normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());
				Point minLoc;
			    minMaxLoc(result, NULL, NULL, &minLoc, NULL, Mat());
			    saveToMap(patch, minLoc, i, j, patchHeight, patchWidth);

			}
		}

		disparityMap.convertTo(disparityMap, CV_8U);
		imshow("Disparity Map", disparityMap);
		waitKey(0);

	}

	void saveToMap(Mat patch, Point location, int i, int j, int patchHeight, int patchWidth){

		Mat one = getLocationMatrix(i, j, patchWidth, patchHeight);
		Mat two = getLocationMatrix(location.x, location.y, patchWidth, patchHeight);

		Mat result;
		absdiff(one, two, result);
		//absdiff(patch, image2(Rect(location.x, location.y, patchWidth, patchHeight)).clone(), result);
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
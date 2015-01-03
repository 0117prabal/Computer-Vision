#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class IntegralImages{

private:

	Mat image;
	Mat grayImage;
	Mat IntegralImage;
	Mat IntegralImageCV;

public:

	IntegralImages(string path){

		image = imread(path, 1);

		if(!image.data){

			cout<<"Unable to read the image"<<endl;
			exit(-1);
		}

		cvtColor(image, grayImage ,CV_RGB2GRAY);

		if(!grayImage.data){

			cout<<"Unable to convert image to gray scale"<<endl;
			exit(-1);
		}		

		grayImage.convertTo(grayImage, CV_32F);
		IntegralImage = calculateIntegralImage();
		integral(grayImage, IntegralImageCV);
	}

	void computeIntensities(){

		vector<Rect> rectangles = getRandomRectangles();
		drawRectanglesAndComputeIntensity(rectangles);
	}

	void drawRectanglesAndComputeIntensity(vector<Rect> rectangles){

		for(int i = 0 ; i < rectangles.size() ; i++){

			rectangle(image, rectangles[i].tl(), rectangles[i].br(), Scalar(0, 0, 255), 1, 8, 0);
			cout<<endl<<"Average intensity manually calculated: "<<averageIntensityManual(rectangles[i])<<endl;
			cout<<"Average Intensity without using integral function: "<<averageIntensityWithoutIntegral(rectangles[i])<<endl;
			cout<<"Average intensity with using integral funcion: "<<averageIntensityWithIntegral(rectangles[i])<<endl;
			imshow("Image", image);
			waitKey(0);
		}

		destroyAllWindows();
	}

	double averageIntensityWithIntegral(Rect rect){

		double sum = getSum(rect, IntegralImageCV);

		return sum / (rect.height * rect.width);
	}

	double averageIntensityWithoutIntegral(Rect rect){

		double sum = getSum(rect, IntegralImage);

		return sum / (rect.height * rect.width);
	}

	Mat calculateIntegralImage(){

		Mat ret = Mat::zeros(grayImage.rows+1, grayImage.cols+1, CV_64F);

		double rowSum = 0.0;

		for(int i = 1 ; i < grayImage.rows ; i++){
			
			rowSum = 0.0;

			for(int j = 1 ; j <grayImage.cols ; j++){

				rowSum += grayImage.at<float>(i-1, j-1);
				ret.at<double>(i,j) = ret.at<double>(i-1, j) + rowSum;
				
			}

		}

		return ret;
	}

	double getSum(Rect rect, Mat IntegralImage){

		double sum(0.0);

		sum = IntegralImage.at<double>(rect.y + rect.height, rect.x + rect.width)
			- IntegralImage.at<double>(rect.y + rect.height, rect.x)
			- IntegralImage.at<double>(rect.y, rect.x + rect.width)
			+ IntegralImage.at<double>(rect.y, rect.x);

		return sum; 
	}

	float averageIntensityManual(Rect rect){

		float avgIntensity = 0.f;

		Mat temp = grayImage(rect).clone();

		for(int i = 0 ; i < temp.rows ; i++){
			for(int j = 0 ; j < temp.cols ; j++){

				avgIntensity += temp.at<float>(i,j);

			}
		}

		return avgIntensity / (temp.rows * temp.cols);
	}

	vector<Rect> getRandomRectangles(){
		
		vector<Rect> rectangles;

		uint64 initValue = time(0);
		RNG random(initValue);

		for(int i = 0 ; i < 100 ; i++){

			int pointX = random.uniform(1,image.cols);
			int pointY = random.uniform(1,image.rows);
			Rect rect(pointX, pointY, random.uniform(pointX+1, image.cols) - pointX, random.uniform(pointY+1, image.rows) - pointY);
			rectangles.push_back(rect);

		}

		return rectangles;
	}

	~IntegralImages(){

		//nothing to do here

	}

};
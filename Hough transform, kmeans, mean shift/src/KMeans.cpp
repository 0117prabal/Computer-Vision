#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class KMeans{

private:

	Mat image;

public:

	KMeans(string path){

		image = imread(path, 1);

		if(!image.data){
			cout<<"Unable to read the image"<<endl;
		}

		imshow("Original", image);
		waitKey(0);

	}

	void run(){

		intensityClustering();
		colorClustering();
		intensityAndCoordinatePositionClustering();
		destroyAllWindows();

	}

	void intensityClustering(){

		Mat imgFloat;
	    image.convertTo(imgFloat, CV_32FC3, (1./255.));

	    Mat imgGray;
	    cvtColor(imgFloat, imgGray, CV_BGR2GRAY);

	    imshow("Original gray", imgGray);
	    waitKey(0);

	    Mat temp, labels, centers;

	    for (int k=2; k<13; k+=2) {

	    	cout<<"K = "<<k<<endl;
	        imgGray.reshape(1, imgGray.total()).copyTo(temp);
	        kmeans(temp, k, labels, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 0.001), 10, KMEANS_PP_CENTERS, centers);

	        for (int i=0; i<temp.rows; ++i) {
	            
	            memcpy(temp.ptr(i), centers.ptr(*labels.ptr(i)), temp.elemSize()*temp.cols);
	        }

	        temp = temp.reshape(imgGray.channels(), imgGray.rows);

	        imshow("kmeans on gray", temp);
	        waitKey(0);
	    }

	    destroyWindow("Original gray");
	    destroyWindow("kmeans on gray");

	}

	void colorClustering(){

		Mat imgFloat;
		image.convertTo(imgFloat, CV_32FC3, (1.f/255.f));

		Mat temp, labels, centers;


		for(int k = 2 ; k < 20 ; k+=2){

			imgFloat.reshape(1, imgFloat.total()).copyTo(temp);
			cout<<"K = "<<k<<endl;
			kmeans(temp, k, labels, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 0.001), 10, KMEANS_PP_CENTERS, centers);


			for (int i=0; i<temp.rows; ++i) {
	            
	            memcpy(temp.ptr(i), centers.ptr(*labels.ptr(i)), temp.elemSize()*temp.cols);
	        }

	        temp = temp.reshape(imgFloat.channels(), imgFloat.rows);

	        imshow("kmeans on color", temp);
	        waitKey(0);
		}

		destroyWindow("kmeans on color");
	}

	void intensityAndCoordinatePositionClustering(){

	    Mat coords(image.size(), CV_32FC2);
	    
	    for (int y=0; y<coords.rows; ++y) {
	        for (int x=0; x<coords.cols; ++x) {

	            coords.at<cv::Vec2f>(y,x) = Vec2f(x, y);
	        }
	    }

	    Mat imgFloat;
	    image.convertTo(imgFloat, CV_32FC3, (1./255.));

	    Mat imgGray;
	    cvtColor(imgFloat, imgGray, CV_BGR2GRAY);

	    coords *= 1./cv::max(coords.rows, coords.cols);
	    Mat temp2[] = {imgGray, coords};
	    Mat flower_gray_coords;
	    merge(temp2, 2, flower_gray_coords); // flower_gray_coords 3 channels

	    Mat temp, labels, centers;

	    for (int k=2; k<11; k+=2) {

	        flower_gray_coords.reshape(1, flower_gray_coords.total()).copyTo(temp);
	        kmeans(temp, k, labels, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.001), 10, cv::KMEANS_PP_CENTERS, centers);
	        for (int i=0; i<temp.rows; ++i) {
	            memcpy(temp.ptr(i), centers.ptr(*labels.ptr(i)), temp.elemSize()*temp.cols);
	        }
	        // convert Nx3 vector to 2d image
	        temp = temp.reshape(flower_gray_coords.channels(), flower_gray_coords.rows);

	        cout << "k = " << k << endl;
	        imshow("kmeans on gray and pixel coordinates", temp);
	        waitKey(0);
	    }
	    destroyWindow("kmeans on gray and pixel coordinates");

	}

	~KMeans(){

		//nothing to do here

	}

};
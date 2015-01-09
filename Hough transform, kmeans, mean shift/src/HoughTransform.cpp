#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class HoughTransform{

private:

	Mat image;

public:

	HoughTransform(string path){

		image = imread(path, 1);

		if(!image.data){

			cout<<"Unable to read image"<<endl;
		}

	}

	void run(){

		detectUsingBuiltInFunction();
		withoutUsingBuiltInFunction();
	}

	void detectUsingBuiltInFunction(){

		Mat image1;
		Mat image2 = image.clone();
		cvtColor(image, image1, COLOR_BGR2GRAY);

		vector<Vec3f> circles;
		HoughCircles(image1, circles, CV_HOUGH_GRADIENT, 2, 40, 20, 11, 2, 18);

		for(int i = 0 ; i < circles.size() ; i++){

			circle(image2, Point(static_cast<int>( round(circles[i][0])), static_cast<int>( round(circles[i][1]))), static_cast<int>( round(circles[i][2])), Scalar(0, 0, 255), 1, 8, 0);
		}

//		imshow("Detected circles using HoughCircles", image2);
//		waitKey(0);
	}

	void withoutUsingBuiltInFunction(){

		Mat image1;
		Mat edges;

		cvtColor(image, image1, COLOR_BGR2GRAY);
		Canny(image1, edges, 300, 400);

		vector<Mat> accum;

		for(int i = 1 ; i < 6 ; i++){

			Mat temp = Mat::zeros(image1.rows/2, image1.cols/2, CV_32S);
			accum.push_back(temp);

		}
		
		edges.convertTo(edges, CV_32F);

		for(int i = 0 ; i < edges.rows ; i++){

			for(int j = 0 ; j < edges.cols ; j++){

				if(edges.at<float>(i,j) == 0.f){

					continue;

				}


				for(int r = 0 ; r < 5 ; r++){

					for(double theta = 0.0; theta < M_PI*2 ; theta += 0.1){

						int a = static_cast<int>(round((i+1) - (r+1)*cos(theta))) / 2.f;
						int b = static_cast<int>(round((j+1) + (r+1)*sin(theta))) / 2.f;

						if (a<0 || b<0 || a>=accum[r].cols || b>=accum[r].rows){

						  	continue; 

						}

						accum[r].at<int>(a,b)++;
					}

				}

			}
		}

		double tmpMin, tmpMax;
		Point tmpMinPt, tmpMaxPt;
		int max_rrr_Idx;
		Point maxPt;
		double maxVal = -999;

		for(int r = 0 ; r < 5 ; r++){

			minMaxLoc(accum[r], &tmpMin, &tmpMax, &tmpMinPt, &tmpMaxPt);

			if (tmpMax > maxVal){

    	    	maxVal = tmpMax;
                maxPt = tmpMaxPt;
                max_rrr_Idx = r;
            }

		}

		Mat accumDISP;
        accum[max_rrr_Idx].convertTo( accumDISP, CV_8UC1, 255/maxVal );
        imshow("image",accumDISP);
        waitKey(0);

	}

	~HoughTransform(){

		//nothing to do here
	}
};
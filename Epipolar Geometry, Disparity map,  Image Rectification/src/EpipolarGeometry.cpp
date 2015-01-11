#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;


class EpipolarGeometry{

private:

	Mat image1, image2;
	vector<Point2f> ImageOnePoints;
	vector<Point2f> ImageTwoPoints;
	Mat fundamentalMatrix;
	Mat T1;
	Mat T2;

public:

	EpipolarGeometry(vector<Point2f> ImageOnePoints, vector<Point2f> ImageTwoPoints){

		this->ImageOnePoints = ImageOnePoints;
		this->ImageTwoPoints = ImageTwoPoints;

	}

	void visualize(){
		
		cout<<endl<<"Part 2"<<endl;
		image1 = imread("images/apt1.jpg", 1);
		image2 = imread("images/apt2.jpg", 1);

		if(!image1.data || !image2.data){

			cout<<"Unable to load the images"<<endl;
			return;

		}

		drawPoints(image1, ImageOnePoints);
		drawPoints(image2, ImageTwoPoints);

		drawEpipolarLines(image2, ImageOnePoints, 1);
		drawEpipolarLines(image1, ImageTwoPoints, 2);

		imshow("Frame 1", image1);
		waitKey(0);

		imshow("Frame 2", image2);
		waitKey(0);

		destroyAllWindows();
	}

	void calculateFundamentalMatrix(){

		vector<Point2f> normalizedImageOnePoints = normalize(ImageOnePoints, 1);
		vector<Point2f> normalizedImageTwoPoints = normalize(ImageTwoPoints, 2);

		Mat A = createMatrixA(normalizedImageTwoPoints, normalizedImageOnePoints);

		Mat w,u,vt;
		SVD::compute(A, w, u, vt, 0);

		transpose(vt,vt);
		Mat temp = vt.col(vt.cols-1);
		fundamentalMatrix = reshape(temp);

		SVD::compute(fundamentalMatrix, w, u, vt, 0);

		Mat w_temp = Mat::zeros(3, 3, CV_32F);
		w_temp.at<float>(0,0) = w.at<float>(0,0); 
		w_temp.at<float>(1,1) = w.at<float>(0,1);

		fundamentalMatrix = (u*w_temp)*vt;

		fundamentalMatrix = T2.t() * fundamentalMatrix * T1;

		cout<<"Fundamental Matrix:"<<endl<<fundamentalMatrix<<endl;
	}

	vector<Point2f> normalize(vector<Point2f> points, int type){

		vector<Point2f> result;

		float x_avg = 0.f, y_avg = 0.f;

		for(int i = 0 ; i < points.size() ; i++){

			x_avg += points[i].x;
			y_avg += points[i].y;

		}

		x_avg /= points.size();
		y_avg /= points.size();

		float d = 0.f;

		for(int i = 0 ; i < points.size() ; i++){

			d += sqrt(pow((points[i].x - x_avg), 2) + pow((points[i].y - y_avg), 2)) / (sqrt(2)*points.size());

		}


		Mat T(3, 3, CV_32F);

		T.at<float>(0,0) = 1/d;
		T.at<float>(0,1) = 0;
		T.at<float>(0,2) = -x_avg/d;
		T.at<float>(1,0) = 0;
		T.at<float>(1,1) = 1/d;
		T.at<float>(1,2) = -y_avg/d;
		T.at<float>(2,0) = 0;
		T.at<float>(2,1) = 0;
		T.at<float>(2,2) = 1;

		for(int i = 0 ; i < points.size() ; i++){

			Point2f point = toPoint2f(T * toMat(points[i]));
			result.push_back(point);

		}

		switch(type){
			
			case 1:
				T1 = T;
				break;
			case 2:
				T2 = T;
				break;
			default:
				cout<<"Invalid type"<<endl;
				break;
		}


		return result;
	}

	Mat toMat(Point2f point){

		Mat temp(3, 1, CV_32F);
		temp.at<float>(0,0) = point.x;
		temp.at<float>(1,0) = point.y;
		temp.at<float>(2,0) = 1.f;
		return temp;
	}

	Point2f toPoint2f(Mat mat){

		Point2f ret;
		ret.x = mat.at<float>(0,0);
		ret.y = mat.at<float>(1,0);
		return ret;
	}

	Mat createMatrixA(vector<Point2f> p1, vector<Point2f> p2){

		assert(p1.size() == p2.size());

		Mat ret(p1.size(), 9, CV_32F);

		for(int i = 0 ; i < p1.size() ; i++){

			ret.at<float>(i,0) = p1[i].x * p2[i].x;
			ret.at<float>(i,1) = p1[i].x * p2[i].y;
			ret.at<float>(i,2) = p1[i].x;
			ret.at<float>(i,3) = p1[i].y * p2[i].x;
			ret.at<float>(i,4) = p1[i].y * p2[i].y;
			ret.at<float>(i,5) = p1[i].y;
			ret.at<float>(i,6) = p2[i].x;
			ret.at<float>(i,7) = p2[i].y;
			ret.at<float>(i,8) = 1.f;

		}

		return ret;
	}

	Mat reshape(Mat mat){

		Mat ret(3, 3, CV_32F);
		int counter = 0;

		for(int i = 0 ; i < 3 ; i++){
			for(int j = 0 ; j < 3 ; j++){

				ret.at<float>(i,j) = mat.at<float>(counter++,0);

			}
		}

		return ret;

	}

	void drawPoints(Mat & image, vector<Point2f> points){

		for(int i = 0 ; i < points.size() ; i++){

			circle(image, points[i], 3, Scalar(0,0,255), 2, 8, 0);

		}

	}

	void drawEpipolarLines(Mat & image, vector<Point2f> points, int type){

		assert(fundamentalMatrix.data != NULL);
		assert(type == 1 || type ==2);

		for(int i = 0 ; i < points.size() ; i++){

			Mat temp = toMat(points[i]);
			
			if(type == 1){

				temp = fundamentalMatrix * temp;

			}
			else{

				temp = temp.t() * fundamentalMatrix;
			}

			float y1 = (-temp.at<float>(0,2)) / temp.at<float>(0,1);
			float y2 = (temp.at<float>(0,0) * image.cols + temp.at<float>(0,2)) / -temp.at<float>(0,1);

			line(image, Point(image.cols-1, y2), Point(0.f, y1), Scalar(255, 0, 0), 1, 8, 0);

		}

	}

	void checkFundamentalMatrix(){

		for(int i = 0 ; i < ImageOnePoints.size() ; i++){

			cout<<toMat(ImageOnePoints[i]).t() * fundamentalMatrix * toMat(ImageTwoPoints[i])<<endl;

		}

	}



	Mat getFundamentalMatrix(){

		return fundamentalMatrix;
	}

	Mat getImage1(){

		return image1;
	}

	Mat getImage2(){

		return image2;
	}

	~EpipolarGeometry(){

		// nothing to do here

	}

};
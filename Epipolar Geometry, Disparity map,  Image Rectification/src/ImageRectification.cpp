#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class ImageRectification{


private:

	Mat ImageLeft;
	Mat ImageRight;
	Mat fundamentalMatrix;

public:

	ImageRectification(){

		ImageLeft = imread("images/apt1.jpg", 1);
		ImageRight = imread("images/apt2.jpg", 1);

		if(!ImageLeft.data || !ImageRight.data){

			cout<<"Unable to read the images"<<endl;

		}
		
		fundamentalMatrix.create(3, 3, CV_32F);

		fundamentalMatrix.at<float>(0,0) = -1.78999e-7; fundamentalMatrix.at<float>(0,1) = 5.70878e-6; fundamentalMatrix.at<float>(0,2) = -0.00260653;
		fundamentalMatrix.at<float>(1,0) = -5.71422e-6; fundamentalMatrix.at<float>(1,1) = 1.63569e-7; fundamentalMatrix.at<float>(1,2) = -0.0068799;
		fundamentalMatrix.at<float>(2,0) =  0.00253316; fundamentalMatrix.at<float>(2,1) = 0.00674493; fundamentalMatrix.at<float>(2,2) =  0.191989;
		

		this->fundamentalMatrix = fundamentalMatrix;
		imshow("original", ImageLeft);
		waitKey(0);
	}

	void rectify(){

		Mat projectiveTransformImage1 = getProjectiveTransformMatrix(1);
		//Mat projectiveTransformImage2 = getProjectiveTransformMatrix(2);
		Mat SimilarityTransform1 = getSimilarityTransform(projectiveTransformImage1);

		Mat result;
		warpPerspective(ImageLeft, result, projectiveTransformImage1, Size(ImageLeft.cols, ImageLeft.rows));
		//warpPerspective(result, result, SimilarityTransform1, Size(ImageLeft.cols, ImageLeft.rows));
		imshow("result", result);
		waitKey(0);
	}

	Mat getProjectiveTransformMatrix(int type){

		assert(type == 1 || type == 2);

		Mat result = Mat::zeros(3, 3, CV_32F);

		Mat epipole = getEpipole(type);	
		Mat A = getA(epipole);
		Mat B = getB(epipole);

		Mat eigenValues, temp;
		eigen(A, eigenValues, temp);
		ensurePositiveDefinite(A, eigenValues.at<float>(2,0));
		Mat D = CholeskyDecomposition(A);
		
		Mat Dt = D.t();

		Mat W,U,Vt;
		SVD::compute(Dt.inv() * B * D.inv(), W, U, Vt);

		Mat y = Vt.t().col(0);
		Mat z = D.inv() * y;

		Mat w = toCrossPoductMatrix(epipole) * z;

		result.at<float>(0,0) = 1.f;
		result.at<float>(1,1) = 1.f;
		result.at<float>(2,2) = 1.f;
		result.at<float>(2,0) = w.at<float>(0,0);
		result.at<float>(2,1) = w.at<float>(1,0);

		return result;
	}

	Mat getSimilarityTransform(Mat p){

		Mat result = Mat::zeros(3, 3, CV_32F);

		result.at<float>(0,0) = fundamentalMatrix.at<float>(2,1) - p.at<float>(2,1) * fundamentalMatrix.at<float>(2,2);
		result.at<float>(0,1) = p.at<float>(2,0) * fundamentalMatrix.at<float>(2,2) - fundamentalMatrix.at<float>(2,0);

		result.at<float>(1,0) = fundamentalMatrix.at<float>(2,0) - p.at<float>(2,0) * fundamentalMatrix.at<float>(2,2);
		result.at<float>(1,1) = fundamentalMatrix.at<float>(2,1) - p.at<float>(2,1) * fundamentalMatrix.at<float>(2,2);
		result.at<float>(1,2) = fundamentalMatrix.at<float>(2,2);

		result.at<float>(2,2) = 1.f;

		return result;
	}

	Mat getEpipole(int type){

		Mat epipole;
		Mat temp(3, 3, CV_32F);

		assert(type == 1 || type == 2);

		if(type==1){

			temp = fundamentalMatrix;

		}

		else{

			temp = fundamentalMatrix.t();
		}

		Mat U, W, Vt;
		SVD::compute(temp, W, U, Vt, 0);
		transpose(Vt, Vt);
		epipole = Vt.col(Vt.cols-1);

		return epipole;
	}

	Mat getA(Mat epipole){

		Mat A(3, 3, CV_32F);
		Mat PPT = Mat::zeros(3, 3, CV_32F);

		PPT.at<float>(0,0) = ((float)(ImageRight.rows * ImageRight.cols) / 12.f) * (pow(ImageRight.rows, 2) - 1.f);
		PPT.at<float>(1,1) = ((float)(ImageRight.rows * ImageRight.cols) / 12.f) * (pow(ImageRight.cols, 2) - 1.f);

		A = (toCrossPoductMatrix(epipole)).t() * PPT * toCrossPoductMatrix(epipole);

		return A;
	}

	Mat getB(Mat epipole){

		Mat B(3, 3, CV_32F);
		Mat PcPcT(3, 3, CV_32F);

		float factor = 1.f/4.f;

		PcPcT.at<float>(0,0) = pow(ImageRight.cols-1, 2); 					   PcPcT.at<float>(0,1) = (ImageRight.cols - 1) * (ImageRight.rows - 1);  PcPcT.at<float>(0,2) = 2*(ImageRight.cols - 1);
		PcPcT.at<float>(1,0) = (ImageRight.cols - 1) * (ImageRight.rows - 1);  PcPcT.at<float>(1,1) = pow(ImageRight.rows - 1 , 2); 	              PcPcT.at<float>(1,2) = 2*(ImageRight.rows - 1);
		PcPcT.at<float>(2,0) = 2*(ImageRight.cols - 1); 					   PcPcT.at<float>(2,1) = 2*(ImageRight.rows - 1); 						  PcPcT.at<float>(2,2) = 4;
		PcPcT = factor * PcPcT;

		B = toCrossPoductMatrix(epipole).t() * PcPcT * toCrossPoductMatrix(epipole);

		return B;
	}

	Mat toCrossPoductMatrix(Mat mat){

		Mat result = Mat::zeros(3, 3, CV_32F);

		result.at<float>(0,1) = -mat.at<float>(2,0);
		result.at<float>(0,2) =  mat.at<float>(1,0);
		result.at<float>(1,0) =  mat.at<float>(2,0);
		result.at<float>(1,2) = -mat.at<float>(0,0);
		result.at<float>(2.0) = -mat.at<float>(1,0);
		result.at<float>(2,1) =  mat.at<float>(0,0);


		return result;
	}

	Mat CholeskyDecomposition(Mat mat){
		
		Mat result= (Mat_<float>(3,3) << 2.5182e1, 0, 3.0017e4, 0, 3.7774e1, 1.5835e4, 0, 0, 1.2349e1);

		return result;
	}

	void ensurePositiveDefinite(Mat &A, float error){

		A.at<float>(0,0) += 0.0001f;
		A.at<float>(1,1) += 0.0001f;
		A.at<float>(2,2) += 0.0001f;

	}

	~ImageRectification(){

		// nothing to do here

	}

};
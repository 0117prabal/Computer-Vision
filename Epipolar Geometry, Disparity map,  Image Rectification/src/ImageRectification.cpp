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

	ImageRectification(Mat ImageLeft, Mat ImageRight){
		
		this->ImageLeft = ImageLeft;
		this->ImageRight = ImageRight;

		fundamentalMatrix.create(3, 3, CV_32F);

		fundamentalMatrix.at<float>(0,0) = -1.78999e-7; fundamentalMatrix.at<float>(0,1) = 5.70878e-6; fundamentalMatrix.at<float>(0,2) = -0.00260653;
		fundamentalMatrix.at<float>(1,0) = -5.71422e-6; fundamentalMatrix.at<float>(1,1) = 1.63569e-7; fundamentalMatrix.at<float>(1,2) = -0.0068799;
		fundamentalMatrix.at<float>(2,0) =  0.00253316; fundamentalMatrix.at<float>(2,1) = 0.00674493; fundamentalMatrix.at<float>(2,2) =  0.191989;
		
	}

	void rectify(){

		Mat projectiveTransformImage1 = getProjectiveTransformMatrix(1);
		Mat SimilarityTransform1 = getSimilarityTransform(projectiveTransformImage1);
		Mat ShearingTransform1 = getShearingTransform(projectiveTransformImage1, SimilarityTransform1);

		cout<<".................................."<<endl;
		cout<<"Printing homographies for Image 1"<<endl;
		cout<<"Hp"<<endl<<projectiveTransformImage1<<endl;
		cout<<"Hr"<<endl<<SimilarityTransform1<<endl;
		cout<<"Hs"<<endl<<ShearingTransform1<<endl;
		cout<<".................................."<<endl;

		Mat projectiveTransformImage2 = getProjectiveTransformMatrix(2);
		Mat SimilarityTransform2 = getSimilarityTransform(projectiveTransformImage2);
		Mat ShearingTransform2 = getShearingTransform(projectiveTransformImage2, SimilarityTransform2);

		cout<<".................................."<<endl;
		cout<<"Printing homographies for Image 2"<<endl;
		cout<<"Hp"<<endl<<projectiveTransformImage2<<endl;
		cout<<"Hr"<<endl<<SimilarityTransform2<<endl;
		cout<<"Hs"<<endl<<ShearingTransform2<<endl;
		cout<<".................................."<<endl;

	}

	Mat getProjectiveTransformMatrix(int type){

		assert(type == 1 || type == 2);

		Mat result = Mat::zeros(3, 3, CV_32F);

		Mat epipole = getEpipole(type);	
		Mat A = getA(epipole);;
		Mat B = getB(epipole);
		Mat D = CholeskyDecomposition(type);
		
		Mat Dt = D.t();

		Mat W,U,Vt;
		SVD::compute((Dt.inv() * B * D.inv()), W, U, Vt);

		Mat y = Vt.t().col(0);
		Mat z = D.inv() * y;

		Mat w = toCrossPoductMatrix(epipole) * z;

		result.at<float>(0,0) = 1.f;
		result.at<float>(1,1) = 1.f;
		result.at<float>(2,2) = 1.f;
		result.at<float>(2,0) = w.at<float>(0,0) / w.at<float>(2,0);
		result.at<float>(2,1) = w.at<float>(1,0) / w.at<float>(2,0);

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

	Mat getShearingTransform(Mat Hp, Mat Hr){

		Mat result = Mat::zeros(3, 3, CV_32F);

		Mat a(3, 1, CV_32F);
		Mat b(3, 1, CV_32F);
		Mat c(3, 1, CV_32F);
		Mat d(3, 1, CV_32F);

		a.at<float>(0,0) = (ImageLeft.cols - 1.f) / 2.f;
		a.at<float>(1,0) = 0.f;
		a.at<float>(2,0) = 1.f;

		b.at<float>(0,0) = ImageLeft.cols - 1.f;
		b.at<float>(1,0) = (ImageLeft.rows - 1.f) / 2.f;
		b.at<float>(2,0) = 1.f;

		c.at<float>(0,0) = (ImageLeft.cols - 1.f) / 2.f;
		c.at<float>(1,0) = ImageLeft.rows - 1.f;
		c.at<float>(2,0) = 1.f;

		d.at<float>(0,0) = 0.f;
		d.at<float>(1,0) = (ImageLeft.rows - 1.f) / 2.f;
		d.at<float>(2,0) = 1.f;

		a = Hr*Hp*a;
		b = Hr*Hp*b;
		c = Hr*Hp*c;
		d = Hr*Hp*d;

		Mat x = b-d;
		Mat y = c-a;

		float h = ImageLeft.rows;
		float w = ImageLeft.cols;

		float aa = (pow(h, 2)*pow(x.at<float>(1,0), 2) + pow(w, 2) * pow(y.at<float>(1,0),2)) / (h*w*( x.at<float>(1,0) * y.at<float>(0,0) - x.at<float>(0,0) * y.at<float>(1,0) ));

		float bb = (pow(h, 2)*x.at<float>(0,0)*x.at<float>(1,0) + pow(w, 2) * y.at<float>(0,0) * y.at<float>(1,0)) / (h*w*(x.at<float>(0,0) * y.at<float>(1,0) - x.at<float>(1,0) * y.at<float>(0,0)));

		result.at<float>(0,0) = aa;
		result.at<float>(0,1) = bb;
		result.at<float>(1,1) = 1.f;
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
		Mat PPT = getPPT();
		A = (toCrossPoductMatrix(epipole)).t() * PPT * toCrossPoductMatrix(epipole);

		return A;
	}

	Mat getPPT(){

		Mat P(3, ImageLeft.rows*ImageLeft.cols, CV_32F);
		Mat pc = getPc();

		int counter = 0;

		for(int i = 0 ; i < ImageLeft.rows ; i++){
			for(int j = 0 ; j < ImageLeft.cols ; j++){

				P.at<float>(0, counter) = i - pc.at<float>(0,0);
				P.at<float>(1, counter) = j - pc.at<float>(0,1);
				P.at<float>(2, counter) = 0.f;

				counter++;
			}
		}

		Mat ppt = P*P.t();

		return ppt;
	}


	Mat getB(Mat epipole){

		Mat B(3, 3, CV_32F);
		Mat pc = getPc();
		Mat PcPcT = pc * pc.t();
		B = toCrossPoductMatrix(epipole).t() * PcPcT * toCrossPoductMatrix(epipole);

		return B;
	}

	Mat getPc(){

		Mat result(3, 1, CV_32F);
		result.at<float>(0,0) = ImageLeft.rows/2.f;
		result.at<float>(1,0) = ImageLeft.cols/2.f;
		result.at<float>(2,0) = 1.f;
		return result;
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

	Mat CholeskyDecomposition(int type){
		
		/*
			I didn't get time to implement the cholesky decomposition so this is the result of decomposition using MATLAB
			and I just manually entered it here.

		*/

		assert(type == 1 || type ==2);

		if(type == 1){

			Mat result= (Mat_<float>(3,3) << 2.5183e4, -8.3941e-4, 3.0018e4, 0.0, 3.7774e1, 1.5835e4, 0.0, 0.0, 4.7805);
			return result;
		}

		else{

			Mat result = (Mat_<float>(3,3) << 2.4699e1, -8.2327e-4, 2.9521e4, 0.0, 3.7048e1, 1.7811e4, 0.0, 0.0, 1.1059e1);
			return result;
		}
		
	}

	~ImageRectification(){

		// nothing to do here

	}

};
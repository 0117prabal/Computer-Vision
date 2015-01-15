#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class GeodesicActiveContours{

private:

	Mat image;

public:

	GeodesicActiveContours(string path){

		image = imread(path, 1);

		if(!image.data){

			cout<<"Unable to read the image"<<endl;
		}

		image.convertTo( image, CV_32FC3, (1./255.));

	}

	void run(){

		Point2i center = Point2i(image.cols/2, image.rows/2);
    	float radius = min(image.cols/3, image.rows/3);
    	Mat phi;
    	levelSetContours(center, radius, phi);
	}

	void levelSetContours(const Point2i& center, const float& radius, Mat& phi){

		phi.create(image.size(), CV_32FC1);

		for(int y=0; y<phi.rows; y++){

		   	const float disty2 = pow(y-center.y, 2);
           	
           	for (int x=0; x<phi.cols; x++){

           		phi.at<float>(y,x) = disty2 + pow(x-center.x, 2);

           	}
        }

        sqrt(phi, phi);
        phi = (radius - phi);
        Mat temp = computeContour(phi, 0.0f);
 		
 		cout << "Press any key to continue...\n" << endl;
	    showGray(phi, "phi", 1);
    	showContour(image, temp, 0);

    	// derivative kernels
    	Vec3f dx(-0.5f,  0.f,  0.5f), dxx( 1.0f, -2.f, 1.0f);
    	Mat phi_x, phi_xx, phi_y, phi_yy, phi_xy, phi_x2, phi_y2, curvature, edges, contour;

    	Mat gx, gy, img_gray;
	    cvtColor(image, img_gray, CV_BGR2GRAY     );
	    sepFilter2D(img_gray, gx, -1, dx, 1.f); // src, dst, ddepth, kernelX, kernelY // Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT
	    sepFilter2D(img_gray, gy, -1, 1.f, dx);

	    Mat w = gx.mul(gx) + gy.mul(gy);
	    sqrt(w, w);
	    w = 1.f/(w + 1.f);

	    double minVal,  maxVal;
	    minMaxLoc( w, &minVal, &maxVal );
	    // stretch the edge response function to 0 to improve convergence against edges
	    w = (w - float(minVal)) * float(maxVal/(maxVal-minVal));

	    // we need the gradients of w to approach towards the edges
    	sepFilter2D(w, gx, -1, dx, 1.f);
    	sepFilter2D(w, gy, -1, 1.f, dx);

    	// estimate timestep
	    const float tau = 1.f/(4.f*maxVal);
	    const float eps = 0.0001f;

	    int i=0;
    	float diff=INFINITY;

    	do{

    		++i;
	        // compute derivatives of phi //    xxx  yyy
	        sepFilter2D(phi,   phi_x,  -1, dx,  1.f );
	        sepFilter2D(phi,   phi_y,  -1, 1.f, dx  );
	        sepFilter2D(phi_x, phi_xy, -1, 1.f, dx  );
	        sepFilter2D(phi,   phi_xx, -1, dxx, 1.f );
	        sepFilter2D(phi,   phi_yy, -1, 1.f, dxx );
	        pow(phi_x, 2, phi_x2);
	        pow(phi_y, 2, phi_y2);

	        curvature  = (phi_xx.mul(phi_y2) - 2.f*phi_x.mul(phi_y.mul(phi_xy)) + phi_yy.mul(phi_x2)).mul(1.f / (phi_x2 + phi_y2 + eps));
     	   	multiply(curvature, w, curvature, tau); // dst = scale * src1 * src2 ////////////////////////////////////////////////////////////////////////////////////////////////
     	   	uphillFrontProp(gx, gy, phi, edges);
     	   	phi += curvature + edges;
     	   	
     	   	if (i%100==0){
	            ////////////////////////////////////
	            contour = computeContour(phi, 0.0f);
	            ////////////////////////////////////
	            showGray(phi, "phi",   1 );
	            showContour( image, contour, 1 );
	            ///////////////////////////////
	            absdiff(contour, temp, temp);
	            diff = sum(temp)[0]/255;
	            std::cout << i << "\t" << diff << " pixels of the contour changed" << std::endl;
	            temp = contour;
	        }

    	} while(i<20000 && diff > 0);

    	cout <<                                   endl;
	    cout << "Press any key to continue..." << endl;
	    cout <<                                   endl;
	    waitKey();
	    destroyWindow("contour");
	    destroyWindow("phi");

	}

	//////////////////////////////////////////////
	// compute the pixels where phi(x,y)==level //
	//////////////////////////////////////////////
	Mat computeContour(const cv::Mat& phi, const float level){

	    CV_Assert( phi.type() == CV_32FC1 );

	    Mat segmented_NORMAL(phi.size(), phi.type());
	    Mat segmented_ERODED(phi.size(), phi.type());

	    threshold(phi, segmented_NORMAL, level, 1.0, THRESH_BINARY);
	    erode(segmented_NORMAL, segmented_ERODED, getStructuringElement(MORPH_ELLIPSE, Size2i(3,3)));

	    return (segmented_NORMAL != segmented_ERODED);
	}

	////////////////////////////
	// show a grayscale image //
	////////////////////////////
	void showGray(const Mat& img, const string title, const int t){
	    
	    assert(img.channels() == 1 );

	    double minVal, maxVal;
	    minMaxLoc(img, &minVal, &maxVal);

	    Mat temp;
	    img.convertTo(temp, CV_32F, 1./(maxVal-minVal), -minVal/(maxVal-minVal));
	    imshow(title, temp);
	    waitKey(t);
	}

	///////////////////////////
	// draw contour on image //
	///////////////////////////
	void showContour(const cv::Mat& img, const cv::Mat& contour, const int t ){

	    assert(img.cols == contour.cols && img.rows == contour.rows && img.type() == CV_32FC3 && contour.type() == CV_8UC1);

	    Mat temp(img.size(), img.type());

	    const Vec3f color(0, 0, 1);

	    for(int y=0; y<img.rows; y++){
	        for (int x=0; x<img.cols; x++){
	            
	            temp.at<Vec3f>(y,x) = contour.at<uchar>(y,x)!=255 ? img.at<cv::Vec3f>(y,x) : color;
	        }
	    }

	    imshow("contour", temp);
	    waitKey(t);
	}

	/////////////////////////////////////////////////////////
	// implementation of the uphill front propagation term //
	/////////////////////////////////////////////////////////
	void uphillFrontProp( const cv::Mat& w_x, const cv::Mat& w_y, const cv::Mat& phi, cv::Mat& result){

	    result.create(w_x.size(), CV_32FC1);

	    for (int y=0; y<result.rows; y++)
	    {   for (int x=0; x<result.cols; x++)
	        {
	            const float& wx = w_x.at<float>(y,x);
	            const float& wy = w_y.at<float>(y,x);
	            const float& p  = phi.at<float>(y,x);

	            result.at<float>(y,x)   =   max(wx, 0.f) * (phi.at<float>(y,min(x+1, result.cols-1))- p) + min(wx,0.f) * (p - phi.at<float>(y,max(x-1,0)))
	                                    +   max(wy, 0.f) * (phi.at<float>(min(y+1, result.rows-1),x) - p) + min(wy,0.f) * (p - phi.at<float>(std::max(y-1,0),x));
	        }
	    }
	}


	~GeodesicActiveContours(){
		// nothing to do here
	}
};
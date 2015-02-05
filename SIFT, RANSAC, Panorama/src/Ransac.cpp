#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

#define IMAGE1 1
#define IMAGE2 2

class Ransac{

private:

	Mat image1;
	Mat image2;

	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;

	Mat descs1;
	Mat descs2;

	vector<DMatch> twoWay;

public:

	Ransac(){

		image1 = imread("images/image1.png", IMREAD_COLOR);
		image2 = imread("images/image2.png", IMREAD_COLOR);
		imshow("Image 1", image1);
		waitKey(0);
		imshow("Image 2", image2);
		waitKey(0);
		destroyAllWindows();

	}

	void run(){

		applySIFT(IMAGE1);
		applySIFT(IMAGE2);

		showKeyPoints(IMAGE1);
		showKeyPoints(IMAGE2);
		destroyAllWindows();
		
		computeNearestMatches();
	}

	void applySIFT(int type){

		assert(type == IMAGE1 || type == IMAGE2);

		SIFT sift;

		if(type == IMAGE1){
			
			sift(image1, noArray(), keypoints1, descs1);

		}
		else{

			sift(image2, noArray(), keypoints2, descs2);
		}

	}

	void showKeyPoints(int type){

		assert(type == IMAGE1 || type == IMAGE2);

		Mat output;

		if(type == IMAGE1){

  			cv::drawKeypoints(image1, keypoints1, output);
			imshow("Keypoints for Image-1", output);

		}
		else{

			cv::drawKeypoints(image2, keypoints2, output);
			imshow("Keypoints for Image-2", output);

		}

		waitKey(0);

	}

	void computeNearestMatches(){
		
		BFMatcher matcher;
	    vector<vector<DMatch> > match12, match21;
	    matcher.knnMatch(descs1, descs2, match12, 2);
	    matcher.knnMatch(descs2, descs1, match21, 2);

	    /// filter matches by ratio test
	    map<int, int> valid12, valid21;
	    const double t=0.4;
	    for (size_t i=0; i<match12.size(); ++i) {
	      if (match12[i][0].distance/match12[i][1].distance <= t){
	        valid12[match12[i][0].queryIdx] = match12[i][0].trainIdx;
	      }
	    }
	    for (size_t i=0; i<match21.size(); ++i) {
	      if (match21[i][0].distance/match21[i][1].distance <= t){
	        valid21[match21[i][0].queryIdx] = match21[i][0].trainIdx;
	      }
	    }

	    /// determine two-way matches
	  for (const auto& m: valid12) {
	    if (valid21[m.second] == m.first){
	      twoWay.push_back(DMatch(m.first, m.second, 0.));
	    }
	  }

	  /// visualize
	  Mat vis;
	  drawMatches(image1, keypoints1, image2, keypoints2, twoWay, vis, Scalar(0,255,0), Scalar(0,0,255), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	  imshow("Matches", vis);
	  waitKey(0);
	  destroyAllWindows();

	}

	~Ransac(){

		//nothing to do here
	}

};
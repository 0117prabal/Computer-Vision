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

		applyRansac();
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

	void applyRansac(){

	  int nSamples = 4;
	  int nIterations = 20;
	  double thresh = 0.1;
	  int minSamples = 4;

	  Mat best_hom;

	  /// RANSAC loop
	  double best_mse = numeric_limits<double>::max();

	  for(int i=0; i<nIterations; i++){

	    /// randomly select some keypoints
	    vector<Point2f> kpts1(nSamples), kpts2(nSamples);
	    for(int j=0; j <nSamples; j++){
	      int idx = rand() % twoWay.size();
	      kpts1[j] = keypoints1[twoWay[idx].queryIdx].pt;
	      kpts2[j] = keypoints2[twoWay[idx].trainIdx].pt;
	    }

	    /// get and apply homography
	    Mat hom = getPerspectiveTransform(kpts2, kpts1);

	    Mat warppedImage2;
	    warpPerspective(image2, warppedImage2, hom,  image1.size());

	    /// calculate inliers and MSE
	    int inliers_count = 0;
	    double total_mse = 0;
	    for(unsigned int j=0; j<keypoints1.size(); j++){

	      int size = keypoints1[j].size;

	      if(size < 1){
	        continue;
	      }

	      int x = keypoints1[j].pt.x - size / 2 ;
	      int y = keypoints1[j].pt.y - size / 2;
	      Rect rect(x,y,size, size);

	      Mat patch1 = image1(rect);
	      Mat patch2 = warppedImage2(rect);

	      Mat diff;
	      absdiff(patch1, patch2, diff);
	      diff.convertTo(diff, CV_32FC3);
	      diff = diff.mul(diff);
	      Scalar s = sum(diff);

	      // calculate MSE and normalize
	      double mse = (s.val[0] + s.val[1] + s.val[2])/(size*size*3*255*255);

	      if(mse < thresh){
	        inliers_count++;
	        total_mse += mse;
	      }
	    }

	    total_mse /= inliers_count;

	    if(inliers_count > minSamples && total_mse < best_mse){
	      best_mse = total_mse;
	      best_hom = hom.clone();
	    }
	  }
	  cout<<"Done."<<endl<<endl;

	  Mat finalWarppedImage2;
	  warpPerspective(image2, finalWarppedImage2, best_hom, Size(image1.cols*2, image1.rows));
	  imshow("Image-1", image1);
	  imshow("Wrapped Image-2", finalWarppedImage2);
	  waitKey(0);
	  destroyAllWindows();

	  ///Merge images
	  Mat mask;
	  cvtColor(finalWarppedImage2, mask, CV_RGB2GRAY );
	  threshold( mask, mask, 0, 1,THRESH_BINARY_INV );

	  Rect roi(0,0, image1.cols, image1.rows);
	  mask = mask(roi);
	  Mat temp = finalWarppedImage2(roi);

	  add(temp, image1, temp, mask);
	  imshow("Wrapped Image-2", finalWarppedImage2);
	  waitKey(0);
	  destroyAllWindows();


	}

	~Ransac(){

		//nothing to do here
	}

};
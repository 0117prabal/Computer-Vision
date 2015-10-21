#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "Group.cpp"

#include <iostream>

using namespace std;
using namespace cv;

class SaliencyMap{

private:

	Mat saliencyImage;

	vector <Mat> featureMapO;
	vector <Mat> featureMapI;
	vector <Mat> featureMapCA;
	vector <Mat> featureMapCB;

	Mat conspicuityMapO;
	Mat conspicuityMapI;
	Mat conspicuityMapC;

public:

	SaliencyMap(){

	}

	// function for across scale addition of images

	Mat acrossScaleAdition(vector <Mat> input){

		Mat result = input[0];


		for(int i = 1 ; i < input.size() ; i++){

			Mat temp;
			resize(input[i], temp, input[0].size());
			result = result + temp;
		}


		return result;
	}

	// add all the oriented pyramids after accross scale addition to a single feature map O vector

	void setFeatureMapO(vector <Mat> or0, vector <Mat> or45, vector <Mat> or90, vector <Mat> or135){

		featureMapO.push_back(acrossScaleAdition(or0));
		featureMapO.push_back(acrossScaleAdition(or45));
		featureMapO.push_back(acrossScaleAdition(or90));
		featureMapO.push_back(acrossScaleAdition(or135));

		//showImages(featureMapO);
	}

	void convertToFloat(vector <Mat> & mats){

		for(int i = 0 ; i < mats.size() ; i++){

			mats[i].convertTo(mats[i], CV_32F);
		}

	}

	// add all the oriented pyramids after accross scale addition to a single feature map I vector

	void setFeatureMapI(vector <Mat> I1, vector <Mat> I2){


		featureMapI.push_back(acrossScaleAdition(I1));
		featureMapI.push_back(acrossScaleAdition(I2));

		//showImages(featureMapI);
	}

	// add all the oriented pyramids after accross scale addition to a single feature map C vector for channel A

	void setFeatureMapCA(vector <Mat> CS, vector <Mat> SC){

		featureMapCA.push_back(acrossScaleAdition(CS));
		featureMapCA.push_back(acrossScaleAdition(SC));

		//showImages(featureMapCA);

	}

	// add all the oriented pyramids after accross scale addition to a single feature map C vector for channel B

	void setFeatureMapCB(vector <Mat> CS, vector <Mat> SC){

		featureMapCB.push_back(acrossScaleAdition(CS));
		featureMapCB.push_back(acrossScaleAdition(SC));

		//showImages(featureMapCB);
	}

	// add all the images in the vector and return the result

	Mat addAll(vector <Mat> &input){


		Mat temp = input[0];// = Mat::zeros(input[0].size(), CV_32F);


		for(int i = 1 ; i < input.size() ; i++){
			temp = temp + input[i];
			//addWeighted(input[0], 0.4, input[1], 0.39, 0.0, temp);

		}
		
		//showFloatImage(temp);
		return temp;

	}

	// add all the orientation images to a single image and return the result

	Mat addOrientations(vector<Mat> v){


		Mat temp1 = v[0];

		for(int i = 1 ; i < v.size() ; i++){

			temp1 = v[i] + temp1;

		}


		return temp1;
	}

	/*function for normalizing*/

	void normalizeMaps(vector <Mat> &v){

		double max;

		for(int i = 0 ; i < v.size() ; i++){

			double maxVal;
			minMaxLoc(v[i], NULL, &maxVal);

			if(maxVal > max){

				max = maxVal;

			}
		}

		for(int i = 0 ; i < v.size() ; i++){

			normalize(v[i], v[i], 0, max, NORM_MINMAX, CV_8U);
		}

	}


	// function for calculating uniqueness weights

	void uniquenessWeighted(vector<Mat> &v){

		for(int i = 0 ; i < v.size() ; i++){

			v[i] = v[i] / sqrt((double)v.size());

		}

	}

	void setConspicuityMaps(){
		
		normalizeMaps(featureMapO);


		uniquenessWeighted(featureMapO);
		uniquenessWeighted(featureMapI);
		uniquenessWeighted(featureMapCA);
		uniquenessWeighted(featureMapCB);



		conspicuityMapO = addOrientations(featureMapO);
		conspicuityMapI =  addAll(featureMapI);
		conspicuityMapC = addAll(featureMapCA) + addAll(featureMapCB);

		vector<Mat> conspicuityMaps;

		conspicuityMaps.push_back(conspicuityMapO);
		conspicuityMaps.push_back(conspicuityMapI);
		conspicuityMaps.push_back(conspicuityMapC);

		setSaliencyMap(conspicuityMaps);
	}

	void showFloatImage(Mat fImage){

		double minVal, maxVal;
		minMaxLoc(fImage, &minVal, &maxVal); //find minimum and maximum intensities
		fImage.convertTo(fImage, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
		imshow("fImage", fImage);
		waitKey(0);

	}

	void setSaliencyMap(vector<Mat> maps){

		saliencyImage = Mat::zeros(maps[0].size(), CV_8U);

		for(int i = 0 ; i < maps.size() ; i++){
			saliencyImage = saliencyImage + maps[i];
		}

	}

	void showImages(vector <Mat> images){

		for(int i = 0 ; i < images.size() ; i++){

			imshow(""+i, images[i]);
			waitKey(0);

		}

	}

	void showSaliencyMap(Size size){

		resize(saliencyImage, saliencyImage, size);
		imshow("Saliency map", saliencyImage);
		waitKey(0);
	}

	int findLocalMaxima(Mat & mat, bool check){

	vector<Group> groups;


	for(int i = 0 ; i < mat.rows ; i++){

		for(int j = 0 ; j < mat.cols ; j++){



			int value = (int)mat.at<uchar>(i,j);

			if(value < 0.38 && check){
				continue;
			}


			if((i-1) >= 0  && (j-1) >= 0){

				if(value < (int)mat.at<float>(i-1, j-1)){
					continue;
				}

			}

			if((i-1) >= 0){

				if(value < (int)mat.at<float>(i-1, j)){
					continue;
				}
			}

			if((i-1) >= 0 && (j+1) < mat.cols){
				
				if(value < (int)mat.at<float>(i-1, j+1)){
					continue;
				}
			}

			if((j-1) >= 0){
				
				if(value < (int)mat.at<float>(i, j-1)){
					continue;
				}
			}

			if((j+1) < mat.cols){
				
				if(value < (int)mat.at<float>(i, j+1)){
					continue;
				}
			}

			if((i+1) < mat.rows && (j-1) >= 0 ){
				
				if(value < (int)mat.at<float>(i+1, j-1)){
					continue;
				}
			}

			if((i+1) < mat.rows){
				
				if(value < (int)mat.at<float>(i+1, j)){
					continue;
				}
			}

			if((i+1) < mat.rows && (j+1) < mat.cols){
				
				if(value < (int)mat.at<float>(i+1, j+1) ){
					continue;
				}
			}


			Mat result(3, 1, CV_32F);
			result.at<float>(0,0) = (float)i;
			result.at<float>(1,0) = (float)j;
			result.at<float>(2,0) = (float)value;

			bool b = false;

			for(int k = 0 ; k < groups.size() ; k++){

				if(groups[k].checkMembership(result)){

					groups[k].add(result);
					b = true;
					break;

				}
			}

			if(!b){

				Group newGroup;
				newGroup.add(result);
				groups.push_back(newGroup);

			}



		}

	}

	return groups.size();
}


	~SaliencyMap(){

	}
};
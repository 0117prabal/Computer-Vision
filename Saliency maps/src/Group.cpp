#ifndef GROUP_CPP
#define GROUP_CPP

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

class Group{

private:

	vector<Mat> elements;

public:

	int value;

	Group(){

		value = -1;
	}

	void add(Mat toAdd){


		elements.push_back(toAdd);
		
		if(value == -1){

			value = (int)toAdd.at<uchar>(2,0);

		}

	}

	bool checkMembership(Mat toCheck){

		if((int)toCheck.at<uchar>(2,0) != value){
			return false;
		}

		for(int i = 0 ; i < elements.size() ; i++){

			int ii = (int)elements[i].at<float>(0,0);
			int jj = (int)elements[i].at<float>(1,0);

			int ti = (int)toCheck.at<float>(0,0);
			int tj = (int)toCheck.at<float>(1,0);

			if(abs(ii-ti) < 2 && abs(jj-tj) < 2){
				return true;
			}
		}

		return false;
	}

	int size(){

		return elements.size();
	}

	void printLocations(){

		for(unsigned int i = 0 ; i < elements.size() ; i++){

			cout<<endl<<"I: "<<(int)elements[i].at<float>(0,0)<<endl;
			cout<<"J: "<<(int)elements[i].at<float>(1,0)<<endl<<endl;
		}

	}

	int getY(){

		return (int)elements[0].at<float>(1,0);
	}

	int getX(){

		return (int)elements[0].at<float>(0,0);
	}

	~Group(){

	}

};

#endif
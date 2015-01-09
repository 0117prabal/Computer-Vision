#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class Parser{

private:

	string filename;
	vector <Point2f> ImageOnePoints;
	vector <Point2f> ImageTwoPoints; 

public:

	Parser(string filename){

		this->filename = filename;

	}

	void parse(){

		ifstream datafile(filename.c_str());

		if(!datafile.is_open()){

			cout<<"Unable to read data"<<endl;
			return;

		}

		string line;
		getline(datafile, line);

		while(getline(datafile, line)){

			stringstream ss(line);
			Point2f p1,p2;

			ss >> p1.x;
			ss >> p1.y;

			ss >> p2.x;
			ss >> p2.y;

			ImageOnePoints.push_back(p1);
			ImageTwoPoints.push_back(p2);
		}

		datafile.close();

	}

	vector<Point2f> getImageOnePoints(){
		return ImageOnePoints;
	}

	vector<Point2f> getImageTwoPoints(){
		return ImageTwoPoints;
	}

	~Parser(){

		// nothing to do here

	}

};
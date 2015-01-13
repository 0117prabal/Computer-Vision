#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Snakes{

  private:

      Mat image;

  public:

  Snakes(string path){

    image = imread(path, 1);

    if(!image.data){

        cout<<"Unable to read data"<<endl;
    }

  }

  void run(){

  }

  ~Snakes(){

    //nothing to do here

  }

};

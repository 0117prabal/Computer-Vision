# Saliency Maps

The general overview of the saliency map computation architecture is shown below.

![arch](https://github.com/musaeed/Computer-Vision/raw/master/Saliency%20maps/examples/architecture.png)

For more details on the saliency maps using this architecture, please refer to the original paper [here](http://www.iai.uni-bonn.de/~frintrop/paper/frintrop_cahb2011.pdf).

## Build and run

To compile the program, you need the Open CV libraries installed on your system. To install this library please follow the instructions [here](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html). After installing the required libraries, go to the directory and use the CMakeLists.txt file to generate the Makefile.

```bash
cmake .
```
Now use this Makefile to compile the sources. The compiled binary will be created inside the bin folder, simply execute it from there. The image path can be passed as a command line argument to the executable to generate the corresponding saliency map.

## Examples

Some example image / saliency map pair are given below which are produced using this source code and can be easily reproduced.

![](https://github.com/musaeed/Computer-Vision/raw/master/Saliency%20maps/examples/baw_saliency.png)
![](https://github.com/musaeed/Computer-Vision/raw/master/Saliency%20maps/examples/pop_saliency.png)
![](https://github.com/musaeed/Computer-Vision/raw/master/Saliency%20maps/examples/uni-bonn_saliency.png)

## Results

I implemented this saliency system as part of a competition held at my university (Uni Bonn). My implementation out-performed all the other implementations done by other students. The Precision-recall curve for some of the best implementations is given below. The curve corresponding to my implementation is Prabal-Muhammad.

![](https://github.com/musaeed/Computer-Vision/raw/master/Saliency%20maps/examples/pr_curve.png)


For any questions please contact me at muhammad.omar555 [at] gmail . com.
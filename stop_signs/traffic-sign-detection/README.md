# Traffic Sign Detection: Setup
Real-time traffic sign detection using Gielis curves

## System & dependencies

System requirements:

* Ubuntu 14.04
* CMake 2.8.12
* Eigen 3.2.0
* OpenCV 2.4.8
* GTest 1.6.0

### CMake Library

`sudo apt-get install cmake`

The version available in Ubuntu 14.04 repositories is CMake 3.2.2

### Eigen Library

`sudo apt-get install libeigen3-dev`

The version available in Ubuntu 14.04 repositories is Eigen 3.2.0

### OpenCV Library

`sudo apt-get install libopencv-dev`

The version available in Ubuntu 14.04 repositories is OpenCV 2.4.8

### Google tests

`sudo apt-get install libgtest-dev`

The version available in Ubuntu 14.04 repositories is GTest 1.6.0

However, this command only downloads the source that you need to compile.

```
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make

# copy or symlink libgtest.a and libgtest_main.a to your /usr/lib folder
sudo cp *.a /usr/lib
```

The version available in Ubuntu 14.04 repositories is GTest 1.6.1

## Compilation

* Create a build folder:

`mkdir build`

* Move to the created folder:

`cd build`

* Create the `MakeFile` via cmake:

`cmake ../src`

* Compile the code to generate the executable:

`make` or `make -j n` where `n` is the number of cores to use for the compilation

* A folder bin will be created at the same level as the build directory.

* In order to run the code:

`../bin/main ../test-videos/video01.avi`

# Autodrive Competition - Code Documentation
main.cpp can be found in **src/apps/main.cpp**

### Code Implementation
The program runs by iterating frame by frame through the video stream using a while(true) statement. Before the loop, the program opens the stream and initializes the **contour** variables. These variables store the detected shape vectors (of points) later. In each loop iteration, the frame is first filtered for red pixels. A blur is applied before passing onto contour extraction. The extracted contours are stored in the main **contour** variable. The program then performs a series of **matrix distortions** to reduce the "noise" (avoid detecting small and irrelevant contours). Next, for each detected contour, the program computes the top-right-most and bottom-left-most points (to draw a rectangle around contour). This rectangle is then stored in a data structure that is passed onto the next loop iteration in order to detect overlap. The reason for this is to further reduce false positives. The algorithm previously detected contours in things like trees, cars, leaves, and other objects. By making sure there is continuous overlap over the same contour over a set number of frames, we can be more confident in that the detected contour is our target of a traffic sign. Finally, the program requires a contour to be in 20 continuous frames to be counted as a sign. Once this happens, the contour is drawn to the screen. 

### Shape Tracker Class
The shape tracker class serves as the data structure that is passed between while-loop iterations.


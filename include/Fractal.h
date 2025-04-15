#ifndef FRACTAL_H
#define FRACTAL_H

#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <thread>
#include "../include/Variable.h"
#include <algorithm>
using namespace cv;
using namespace std;
using namespace Variables;

class Fractal { // the fractal class, which describes the functions of generating and rendering Fractal images
private:
	int m_Max = -9999;                       // maximum
	vector<vector<int>> m_SummArray;         // the image matrix
	vector<vector<vector<int>>> m_My_Array;  // the image matrix with streams
	vector<vector<double>> m_real_array; // array that store the trajectories of points in the complex plane as they iterate through a dynamical system real
	vector<vector<double>> m_imaginary_array; // array that store the trajectories of points in the complex plane as they iterate through a dynamical system imaginary

	random_device m_rd;  // random numbers
	mt19937 m_gen;       //  generator random numbers
	uniform_real_distribution<double>
		m_dis;  // uniform distribution of real numbers

	double m_real = 0.0; // A complex number z can be written as z = x + iy. the real part of z(Re(z)). 
	double m_imaginary = 0.0; // A complex number z can be written as z = x + iy,  y is the imaginary part of z(Im(z)). i is the imaginary unit(i ^ 2 = -1).
	int m_index = 0;
	double m_scaling_factor = pow(RANGE_LIMIT_IMAGE, DEGREE_TO_CALCULATE) / LENGHT; // The variable c is a scaling factor, and is the pixel size, calculated as positive_range_limit_image*2 / Length, where Length is the number of pixels in the x-axis
	int m_roundx = 0; // Row index of the pixel.
	int m_roundy = 0; // Column index of the pixel.

public:
	Fractal(); // the default class constructor
	void generateTrajectoryPoint(int threadId, int& index); // The declaration is a function that generates the trajectory of a point
	void mapEscapedTrajectory(int threadId, int index, vector<vector<vector<int>>>& My_Array); // 
	void generateTrajectories(int threadId); // declaring a trajectory generation function
	void accumulateResults(); // declaring a result counting function
	void findMax(); // declaring the maximum search function
	void renderImage(); // declaring an image rendering function
	void generateFractal(); // declaring the image generation function
};
#endif
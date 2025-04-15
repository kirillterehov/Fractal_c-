#include "../include/Fractal.h"
using namespace Variables;

Fractal::Fractal()
	: m_SummArray(LENGHT, vector<int>(WIDTH, 0)),
	m_My_Array(THREAD, vector<vector<int>>(LENGHT, vector<int>(WIDTH, 0))),
	m_real_array(THREAD, vector<double>(SIZE_ARRAYS_OF_TRAJECTORY, 0.0)),
	m_imaginary_array(THREAD, vector<double>(SIZE_ARRAYS_OF_TRAJECTORY, 0.0)),
	m_gen(m_rd()),
	m_dis(-RANGE_LIMIT_IMAGE, RANGE_LIMIT_IMAGE) { // default constructor initialization list. arrays are an image matrix, initialize the streaming image matrix with zeros. generator random numbers initialize random numbers. uniform distribution of real numbers initialize from the negative to the positive limit of the image. arrays of the real and imaginary parts of a complex number are initialized to 0
}


void Fractal::generateTrajectoryPoint(int threadId, int& index) { // a function that generates the trajectory of a point
	for (int j = 0; j < SIZE_ARRAYS_OF_TRAJECTORY - 1; j++) { // This is the main loop that iterates over the points in the trajectory.
		m_real_array[threadId][j + 1] =
			pow(m_real_array[threadId][j], DEGREE_TO_CALCULATE) - pow(m_imaginary_array[threadId][j], DEGREE_TO_CALCULATE) + m_real; // These are the core iteration equations. They calculate the real and imaginary parts of the next point in the trajectory, (j + 1), based on the current point, j
		m_imaginary_array[threadId][j + 1] = int(RANGE_LIMIT_IMAGE) * (m_real_array[threadId][j] * m_imaginary_array[threadId][j]) + m_imaginary; // These are the core iteration equations. They calculate the real and imaginary parts of the next point in the trajectory, (j + 1), based on the current point, j
		if (m_real_array[threadId][j + 1] < -RANGE_LIMIT_IMAGE || m_real_array[threadId][j + 1] > RANGE_LIMIT_IMAGE ||
			m_imaginary_array[threadId][j + 1] > RANGE_LIMIT_IMAGE || m_imaginary_array[threadId][j + 1] < -RANGE_LIMIT_IMAGE) { // This is the “escape condition.” It checks if the magnitude of the (j+1)-th point in the trajectory is greater than range_limit_image (meaning the point is further than range_limit_image units from the origin in any direction)
			index = j + 1; // If the point escapes, the break statement exits the inner for loop.
			break;
		}
	}
}

void Fractal::mapEscapedTrajectory(int threadId, int index, vector<vector<vector<int>>>& My_Array) { // the logic for mapping the trajectory points to pixel coordinates and incrementing the My_Array
	if ((m_real_array[threadId][index] < -RANGE_LIMIT_IMAGE || m_real_array[threadId][index] > RANGE_LIMIT_IMAGE || m_imaginary_array[threadId][index] < -RANGE_LIMIT_IMAGE || m_imaginary_array[threadId][index] > RANGE_LIMIT_IMAGE) && index >= MINIMUM_ITERATIONS) { // This checks if the trajectory took at least 50 steps before escaping. This is a threshold. If it escaped within a very few iterations, it will not be used to create the fractal, and to avoid that the image is mostly filled with similar pixels and reduce the quality
		for (int points = 1; points <= index; points++) { // This loop iterates through the points of the trajectory up to the point where it escaped (index). The variable q represents the step number or iteration number in the trajectory.
			m_roundx = round(m_real_array[threadId][points] / m_scaling_factor) + (LENGHT / RANGE_LIMIT_IMAGE); // These lines are the heart of the mapping process. They convert the complex coordinates (x, y) of each point in the trajectory to pixel coordinates (roundx, roundy).
			m_roundy = round(m_imaginary_array[threadId][points] / m_scaling_factor) + (WIDTH / RANGE_LIMIT_IMAGE); // These lines are the heart of the mapping process. They convert the complex coordinates (x, y) of each point in the trajectory to pixel coordinates (roundx, roundy).
			if (m_roundx < LENGHT && m_roundx >= BEYOND_THE_LOWER_BOUNDARY_FIELD && m_roundy < WIDTH && m_roundy >= BEYOND_THE_LOWER_BOUNDARY_FIELD) { // This crucial check ensures that the calculated pixel coordinates (roundx, roundy) are within the bounds of the image (i.e., they are valid row and column indices). 
				My_Array[threadId][m_roundx][m_roundy]++; // This is where the actual “drawing” happens. It increments the counter for the pixel at the calculated coordinates (roundx, roundy) within the My_Array
			}
		}
	}
}

void Fractal::generateTrajectories(int threadId) { // trajectory generation function all points
	for (int i = 0; i < NUMBER_OF_POINTS / THREAD; i++) {
		m_real = m_dis(m_gen);  // generating a random number using gen as a source of randomness
		m_imaginary = m_dis(m_gen);  // generating a random number using gen as a source of randomness

		generateTrajectoryPoint(threadId, m_index); // challenge a function that generates the trajectory of a point
		mapEscapedTrajectory(threadId, m_index, m_My_Array); // challenge the logic for mapping the trajectory points to pixel coordinates and incrementing the My_Array
	}
}

void Fractal::accumulateResults() { // a function that counts each element of the matrix by adding elements in the same place in each stream
	for (int t = 0; t < THREAD; t++) {
		for (int i = 0; i < LENGHT; i++) {
			for (int j = 0; j < WIDTH; j++) {
				m_SummArray[i][j] += m_My_Array[t][i][j];
			}
		}
	}
}

void Fractal::findMax() { // the maximum search function in the array that we have calculated 
	for (int i = 0; i < LENGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			m_Max = max(m_Max, m_SummArray[i][j]);
		}
	}
}

void Fractal::renderImage() { // image rendering function
	double s;
	s = (MAX_INTENSITY / m_Max); // 255 the maximum color value in the rgb palette
	Mat image(WIDTH, LENGHT, CV_8UC1);  // CV_8U: 8-bit unsigned integer (grayscale values) C1: 1 channel(grayscale)
	for (int i = 0; i < LENGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			double intensity = (m_SummArray[i][j] * s); // This line calculates the intensity value for each pixel. It multiplies the value in SummArray[i][j] by the scaling factor s
			image.at<uchar>(i, j) =
				static_cast<uchar>(intensity);  // accesses the pixel at row i and column j in the image. The <uchar> template argument specifies that the pixel value is an 8-bit unsigned integer. and converts the intensity value(which is likely a double) to an 8 - bit unsigned integer.
		}
	}
	imwrite("buddhabrot_opencv.bmp", image);  // write image
}

void Fractal::generateFractal() { // challenge the image generation function
#pragma omp parallel num_threads(Thread) // This is the OpenMP directive that instructs the compiler to create a parallel region. Specifies that the code within the following block should be executed in parallel by multiple threads.
	{
		int threadId = omp_get_thread_num(); // Inside the parallel region, this line retrieves the ID of the current thread. Each thread will have a unique threadId ranging from 0 to Thread - 1.
		generateTrajectories(threadId);  // This line calls your generateTrajectories function, passing in the threadId as an argument. This is where the thread-specific work (generating trajectories) is performed.
	}
accumulateResults(); // a function that counts each element of the matrix by adding elements in the same place in each stream
findMax(); // challenge the maximum search function in the array that we have calculated
renderImage(); // challenge image rendering function
}
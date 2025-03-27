#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <random>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Fractal {
private:
    int Length = 500; // длина изображения
    int Width = 500; // ширина изображения
    int Thread = 8; //количество потоков
    int max = -9999; // макисмум

    vector<vector<int>>SummArray; // матрица
    vector<vector<vector<int>>>My_Array; // потоковая матрица
    vector<vector<double>>x;
    vector<vector<double>>y;

    random_device rd; // случайные числа
    mt19937 gen; //  генератор случайных чисел
    uniform_real_distribution<double> dis; //равномерное распределение реальных чисел

public:
    Fractal() :
        SummArray(Length, vector<int>(Width, 0)),
        My_Array(Thread, vector<vector<int>>(Length, vector<int>(Width, 0))),
        x(Thread, vector<double>(1000, 0.0)),
        y(Thread, vector<double>(1000, 0.0)), gen(rd()), dis(-2.0, 2.0)
    {
    }
    void generateTrajectories(int threadId) {

        double a, b;
        int index = 0;
        double c = 4.0 / Length;
        int roundx;
        int roundy;

        for (int i = 0; i < 1000000 / Thread; i++) {
            a = dis(gen); //  генерация случайного числа, используя gen в качестве источника случайности
            b = dis(gen);

            x[threadId][0] = 0.0;
            y[threadId][0] = 0.0;

            for (int j = 0; j < 999; j++) {
                x[threadId][j + 1] = pow(x[threadId][j], 2) - pow(y[threadId][j], 2) + a;
                y[threadId][j + 1] = 2 * (x[threadId][j] * y[threadId][j]) + b;
                if (x[threadId][j + 1] < -2 || x[threadId][j + 1] > 2 || y[threadId][j + 1] > 2 || y[threadId][j + 1] < -2) {
                    index = j + 1;
                    break;
                }
            }

            if ((x[threadId][index] < -2 || x[threadId][index] > 2 || y[threadId][index] < -2 || y[threadId][index] > 2) && index >= 50) {
                for (int q = 1; q <= index; q++) {
                    roundx = round(x[threadId][q] / c) + (Length / 2);
                    roundy = round(y[threadId][q] / c) + (Width / 2);
                    if (roundx < Length && roundx >= 0 && roundy < Width && roundy >= 0) {
                        My_Array[threadId][roundx][roundy]++;
                    }
                }
            }
        }
    }

    void accumulateResults() {
        for (int t = 0; t < Thread; t++) {
            for (int i = 0; i < Length; i++) {
                for (int j = 0; j < Width; j++) {
                    SummArray[i][j] += My_Array[t][i][j];
                }
            }
        }
    }

    void find_max() {
        for (int i = 0; i < Length; i++) {
            for (int j = 0; j < Width; j++) {
                if (max < SummArray[i][j]) {
                    max = SummArray[i][j];
                }
            }
        }
    }


    void renderImage() {
        double s;
        s = (255.0 / max);
        Mat image(Width, Length, CV_8UC1); //
        for (int i = 0; i < Length; i++) {
            for (int j = 0; j < Width; j++) {
                double intensity = (SummArray[i][j] * s);
                image.at<uchar>(i, j) = static_cast<uchar>(intensity); // Assign to the image
            }
        }
        imwrite("buddhabrot_opencv.bmp", image); // отрисовка изображения
    }


    void generateFractal() {

#pragma omp parallel num_threads(Thread)
        {
            int threadId = omp_get_thread_num();
            generateTrajectories(threadId);  // Thread-specific work
        }
    accumulateResults();
    find_max();
    renderImage();
    }
};

int main() {

    Fractal fractal;
    fractal.generateFractal();


    return 0;
}
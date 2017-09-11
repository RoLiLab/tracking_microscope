#ifndef BASE_H
#define BASE_H

#define PI	3.1415926535

#include <windows.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <cstdint>
#include <thread>
#include <vector>
#include <random>
#include <math.h>
#include <cmath>
#include <thread>
#include <functional>   // for std::bind
#include <mutex>
#include <math.h>
#include <memory>
#include <fstream>      // std::ifstream
#include <algorithm> 
#include <iostream>
#include <atomic>


typedef uint64_t uint64;
typedef uint32_t uint;
typedef uint16_t uint16;
typedef uint8_t uint8;

using namespace std;

uint64 GetTickCountMillis(void);
double GetCycleCountSeconds(void);
void SetThreadName(const std::string & strThreadName);

class imSize {
public:
	imSize(int _height, int _width);
	imSize(void);
	~imSize(void);
	int height;
	int width;
};


class Point2d {
public:
	Point2d(double _x, double _y);
	Point2d(void);
	~Point2d(void);
	double cross(Point2d);
	double dot(Point2d);
	double norm(void);
	double norm(Point2d b);
	double x;
	double y;
	Point2d operator+ (const Point2d &other) const;
	Point2d operator- (const Point2d &other) const;
	Point2d operator* (const double a) const;
	Point2d operator+ (const double a) const;
};
class Point2i {
public:
	Point2i(int _x, int _y);
	Point2i(void);
	~Point2i(void);
	int cross(Point2i);
	Point2i operator+ (const Point2i &other) const;
	Point2i operator- (const Point2i &other) const;
	Point2i operator* (const int a) const;
	int x;
	int y;
};

class Point2f {
public:
	Point2f(float _x, float _y);
	Point2f(void);
	~Point2f(void);
	int cross(Point2f);
	Point2f operator+ (const Point2f &other) const;
	Point2f operator- (const Point2f &other) const;
	Point2f operator* (const float a) const;
	float x;
	float y;
};

double norm_Point2d(Point2d a);
void createGaussianFilter(float sigma, int length, float * gKernel);


class img_uint8{
public:
	img_uint8(void);
	img_uint8(uint8_t * _data, imSize _imgSize);
	img_uint8(uint8_t * _data, imSize _imgSize, double option);
	~img_uint8(void);
	uint8_t * data;
	imSize imgSize;
	int _datasize;
};


class img_uint16 {
public:
	img_uint16(void);
	img_uint16(uint16_t * _data, imSize _imgSize);
	~img_uint16(void);
	uint16_t * data;
	imSize imgSize;
	int _datasize;
};

class img_float {
public:
	img_float(void);
	img_float(float * _data, imSize _imgSize);
	~img_float(void);
	float * data;
	imSize imgSize;
};
void SleepThread(int Millis);
void SleepThread(float fMillis);
double getfixeddecimal(double a, int n);
// -----------------------------------------

class handshake_startstop {
public:
	handshake_startstop(void);
	~handshake_startstop(void);
	volatile bool b_start;
	volatile bool b_stop;
};
#endif //  BASE_H

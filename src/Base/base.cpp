#include "base.h"
// -----------------------------------------
Point2d::Point2d(double _x, double _y) {
	x = _x;
	y = _y;
}
Point2d::Point2d(void) {
	x = 0;
	y = 0;
}
double Point2d::cross(Point2d b) {
	return x*b.y - y * b.x;
}
double Point2d::dot(Point2d b) {
	return x*b.x + y * b.y;
}

double Point2d::norm(Point2d b) {
	return sqrt((x - b.x)*(x - b.x) + (y - b.y)*(y - b.y));
}

double Point2d::norm(void) {
	return sqrt(x*x + y*y);
}
Point2d::~Point2d(void) {
}
Point2d Point2d::operator+ (const Point2d &other) const
{
	Point2d result(x + other.x, y + other.y);
	return result;
}

Point2d Point2d::operator- (const Point2d &other) const
{
	Point2d result(x - other.x, y - other.y);
	return result;
}

Point2d Point2d::operator* (const double a) const {
	Point2d result(x*a, y*a);
	return result;
}
Point2d Point2d::operator+ (const double a) const {
	Point2d result(x+a, y+a);
	return result;
}
// -----------------------------------------
Point2i::Point2i(int _x, int _y) {
	x = _x;
	y = _y;
}
Point2i::Point2i(void) {
	x = 0;
	y = 0;
}
Point2i::~Point2i(void) {
}
Point2i Point2i::operator+ (const Point2i &other) const
{
	Point2i result(x + other.x, y + other.y);
	return result;
}

Point2i Point2i::operator- (const Point2i &other) const
{
	Point2i result(x - other.x, y - other.y);
	return result;
}

Point2i Point2i::operator* (const int a) const {
	Point2i result(x*a, y*a);
	return result;
}


int Point2i::cross(Point2i b) {
	return x*b.y - y * b.x;
}

// -----------------------------------------

Point2f::Point2f(float _x, float _y) {
	x = _x;
	y = _y;
}
Point2f::Point2f(void) {
	x = 0;
	y = 0;
}
Point2f::~Point2f(void) {
}
Point2f Point2f::operator+ (const Point2f &other) const
{
	Point2f result(x + other.x, y + other.y);
	return result;
}

Point2f Point2f::operator- (const Point2f &other) const
{
	Point2f result(x - other.x, y - other.y);
	return result;
}

Point2f Point2f::operator* (const float a) const {
	Point2f result(x*a, y*a);
	return result;
}


int Point2f::cross(Point2f b) {
	return x*b.y - y * b.x;
}

// -----------------------------------------
imSize::imSize(void) {
	width = 0;
	height = 0;
}
imSize::imSize(int _width, int _height) {
	width = _width;
	height = _height;
}
imSize::~imSize(void) {
};

// -----------------------------------------

void SleepThread(float fMillis) {
	// Just use the std library
	std::this_thread::sleep_for(std::chrono::microseconds((int)(fMillis*1000.0f)));
} // end Util::SleepThread()
void SleepThread(int Millis) {
	SleepThread((float)Millis);
} // end Util::SleepThread()


uint64 GetTickCountMillis() {
	return (uint64)(GetTickCount());
} // end GetTickCountMillis()

double GetCycleCountSeconds() {

	// Get our frequency and counter, return 0 on failure
	LARGE_INTEGER curTime, freq;
	if (!QueryPerformanceCounter(&curTime)) { return 0.0; }
	if (!QueryPerformanceFrequency(&freq)) {return 0.0; }
	return (double)(curTime.QuadPart) / (double)(freq.QuadPart);
} // end GetCycleCountSeconds()

void SetThreadName(const std::string& strThreadName) {

	// Taken from MSDN: http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
	struct THREADNAME_INFO {
		unsigned long dwType;		// must be 0x1000
		const char* szName;			// pointer to name (in user addr space)
		unsigned long dwThreadID;	// thread ID (-1=caller thread)
		unsigned long dwFlags;		// reserved for future use, must be zero
	};
	struct THREADNAME_INFO info;
	info.dwType = 0x1000;
	info.szName = strThreadName.c_str();
	info.dwThreadID = unsigned long(-1);
	info.dwFlags = 0;
	__try {
		RaiseException(0x406D1388, 0, sizeof(info) / sizeof(ULONG_PTR), reinterpret_cast<ULONG_PTR*>(&info));
	}
	__except (EXCEPTION_EXECUTE_HANDLER) { }

	auto id = std::this_thread::get_id();
	uint64_t* ptr = (uint64_t*)&id;

	char outputFilename[1024];
	sprintf(outputFilename, "d:\\TrackingMicroscopeData\\_report\\%s_thread.txt", strThreadName.c_str());
	FILE  * ofp = fopen(outputFilename, "w");
	fprintf(ofp, "%s thread: %u\n", strThreadName.c_str(), (unsigned int)(*(ptr + 1)));
	fclose(ofp);
} // end SetThreadName()

// -----------------------------------------	

void createGaussianFilter(float sigma, int len, float * gKernel)
{
	
	if (len % 2 == 0) // should be odd number
		return;

	// sum is for normalization
	float sum = 0.0;
	int radius = int(len / 2);
	
	float twoRadiusSquaredRecip = 1.0 / (2.0 * sigma * sigma);
	float sqrtTwoPiTimesRadiusRecip = 1.0 / (sqrt(2.0 * PI) * sigma);

	int r = -radius;
	for (int i = 0; i < len; i++)
	{
		float x = i - radius;
		gKernel[i] =  sqrtTwoPiTimesRadiusRecip * exp(-x * x * twoRadiusSquaredRecip);
		sum += gKernel[i];
	}

	for (int i = 0; i < len; i++)
	{
		gKernel[i] /= sum;
	}
}

double norm_Point2d(Point2d a) {
	return sqrt(a.x*a.x + a.y*a.y);
}

// -----------------------------------------
img_uint8::img_uint8(void) {
	data = NULL;
	_datasize = 0;
}
img_uint8::img_uint8(uint8_t * _data, imSize _imgSize) {
	data = _data;
	imgSize = _imgSize;
	_datasize = _imgSize.height*_imgSize.width*sizeof(char);
}

img_uint8::img_uint8(uint8_t * _data, imSize _imgSize, double option) {
	data = _data;
	imgSize = _imgSize;
	_datasize = (int)(_imgSize.height*_imgSize.width*option*sizeof(char));
}

img_uint8::~img_uint8(void) {
};

// -----------------------------------------
img_uint16::img_uint16(void) {
	data = NULL;
	_datasize = 0;
}
img_uint16::img_uint16(uint16_t * _data, imSize _imgSize) {
	data = _data;
	imgSize = _imgSize;
	_datasize = _imgSize.height*_imgSize.width*sizeof(uint16);
}
img_uint16::~img_uint16(void) {
};

img_float::img_float(void) {
	data = NULL;
}
img_float::img_float(float * _data, imSize _imgSize) {
	data = _data;
	imgSize = _imgSize;
}
img_float::~img_float(void) {
};

// -----------------------------------------
handshake_startstop::handshake_startstop(void) {
	b_start = false;
	b_stop = false;
}
handshake_startstop::~handshake_startstop(void) {
};


// -----------------------------------------
double getfixeddecimal(double a, int n) {
	int i = 0;
	double gain = 1;
	double addition = 0.0000000001;
	switch (n) {
	case 0:
		break;
	case 1:
		gain = 10;
		break;
	case 2:
		gain = 100;
		break;
	case 3:
		gain = 100;
		break;
	case 4:
		gain = 1000;
		break;
	default:
		for (i = 0; i < n; i++)
			gain = gain*10;
	}
	return floor(a*gain) / gain + addition;
}
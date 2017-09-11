#ifndef FlyCapture2_H
#define FlyCapture2_H
#include <string>
#include "FlyCapture2.h"

class PointGreyCamera
{
public:
    PointGreyCamera(void);
    ~PointGreyCamera();
	// stop and start
	volatile bool b_stop;
	volatile bool b_start;
	// -------------
	FlyCapture2::Camera cam;
	FlyCapture2::Image rawImage;
	char errMsg[1024];

	unsigned int getSerialNumber(void);
	void setSerialNumber(unsigned int _serial);

	void setROI(unsigned int _OffsetX, unsigned int _OffsetY, unsigned int _Width, unsigned int _Height);
	void setExpTime(float _ExpTime);

	bool connectCamera(void);
	bool disconnectCamera(void);
	bool releaseCamera(void);

	bool setConfigurations(void);
	bool startCapture(void);
	void resetcount(void);
	bool stopCapturing(void);
	bool grabImage(void);
	uint16 * getDataPointer(void);
	unsigned int getrowBytes(void);
	unsigned int getrows(void);
	unsigned int getcols(void);
	unsigned int getFrameCount(void);
	FlyCapture2::Image getImageCopy(void);
	unsigned int ROIWidth;
	unsigned int ROIHeight;
	unsigned int ROIsz;
	unsigned int ROIsz_byte;
	double bytePerPx;


	FlyCapture2::Error error;
	FlyCapture2::FC2Version fc2Version;
	FlyCapture2::PGRGuid guid;
	FlyCapture2::BusManager busMgr;
	FlyCapture2::CameraInfo NIRCameraInfo;
	FlyCapture2::TriggerMode mTriggerMode;
	FlyCapture2::Property propShutter;
	FlyCapture2::Property propGain;
	FlyCapture2::StrobeControl mStrobe;
	FlyCapture2::Format7ImageSettings mImageSetting;
	FlyCapture2::ImageMetadata metadata;
	FlyCapture2::FC2Config mCamearConfig;
	FlyCapture2::PixelFormat mPixelFormat;
	FlyCapture2::FrameRate mFrameRateMode;


	FlyCapture2::Property propGamma;
	FlyCapture2::Property propExp;
	FlyCapture2::Property propBrightness;

	unsigned int droppedFrames;

private:
	unsigned int SerialNoNIRCamera;
	float ExposureTime;
	unsigned int ROIoffsetX;
	unsigned int ROIoffsetY;
	bool CaptureEnabled;
	unsigned int FrameCount;
	unsigned int InitFrameCount;

	void errorMsgDisp(void);
	void getCameraInfo(void);
	bool setExtTriggerMode(bool _enable);
	bool setShutterSpeed(void);
	bool setGain(float);
	bool setFrameRate(float fps);
	bool setOutputLine(bool _enable, float _Pulsewidth);
	bool setEmbeddedImagePropertyData(void);
	bool setROICamera(void);
	bool setTriggerOff(void);
	bool setFC2Config(void);

	enum{FallingEdge, RisingEdge};
};

#endif // HDF5IMAGEREADER_H

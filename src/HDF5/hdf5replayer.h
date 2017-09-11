#ifndef HDF5REPLAYER_H
#define HDF5REPLAYER_H

#include <string>
#include <H5Cpp.h>
#include <H5Exception.h>
#include "hdf5imagereader.h"
#include "hdf5datareader.h"
#include "hdf5.h"
#include "hdf5_hl.h"

using namespace std;
using namespace H5;

class HDF5Replayer
{
public:
	HDF5Replayer();
	~HDF5Replayer();
	HDF5ImageReader * imagereader_NIR;
	HDF5ImageReader * imagereader_FLR;
	HDF5DataReader * datareader;
	int64_t totalframe_NIR;
	int64_t totalframe_FLR;
	int64_t curframe_NIR;
	int64_t curframe_FLR;
	int64_t stepframe;
	int64_t offsetframe_FLR;
	bool master_NIR; // master -> nir image frame number | false : master -> fluorescent image frame number
	int64_t samplingtime_NIRus;
	int64_t samplingtime_FLRus;
	int64_t updatingtime_us;
	int64_t currentplaytime_us;
	int64_t currentplaytime_usMax;
	bool openNIR(const char * hdf5_filename);
	bool openFLR(const char * hdf5_filename);
	bool openDATA(const char * hdf5_filename);
	img_uint8 curNIRimg;
	img_uint16 curFLRimg;
	void update(void);
	void updateNIR_add(int frameDiff);
	void updateNIR(INT64 frame);
	void updateNIR_time(INT64 _time_us);
	void updateFLR_add(int frameDiff);
	void updateFLR(INT64 frame);
	void updateFLR_time(INT64 _time_us);
	bool b_playing;
private:
};



#endif // HDF5REPLAYER_H

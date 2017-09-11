#include "Base/base.h"
#include "hdf5replayer.h"


HDF5Replayer::HDF5Replayer(void)
{
	imagereader_NIR = NULL;
	imagereader_FLR = NULL;
	datareader = NULL;
	totalframe_NIR = 0;
	totalframe_FLR = 0;
	curframe_NIR = 0;
	curframe_FLR = 0;
	stepframe = 5;
	offsetframe_FLR = 0;
	master_NIR = true; // master -> nir image frame number | false : master -> fluorescent image frame number
	samplingtime_NIRus = 4000;
	samplingtime_FLRus = 8621;
	currentplaytime_us = 0;
	currentplaytime_usMax = 0;
	updatingtime_us = 20000;
	b_playing = false;
}
HDF5Replayer::~HDF5Replayer()
{
	if (imagereader_NIR)
		delete imagereader_NIR;
	if (imagereader_FLR)
		delete imagereader_FLR;
	if (datareader)
		delete datareader;
}


bool HDF5Replayer::openNIR(const char * hdf5_filename){
	if (imagereader_NIR != NULL)
		delete(imagereader_NIR);
	imagereader_NIR = new HDF5ImageReader(hdf5_filename);
	if (!imagereader_NIR)
		return false;
	totalframe_NIR = imagereader_NIR->n_images();
	currentplaytime_usMax = totalframe_NIR*samplingtime_NIRus;
	currentplaytime_us = 0;
	curframe_NIR = 0;
	curNIRimg.imgSize.height = imagereader_NIR->image_height();
	curNIRimg.imgSize.width = imagereader_NIR->image_width();
	if (curNIRimg.data) free(curNIRimg.data);
	curNIRimg.data = (uint8 *)malloc(curNIRimg.imgSize.height * curNIRimg.imgSize.width * sizeof(uint8));
	imagereader_NIR->read(0, curNIRimg.data);
	return true;
}
bool HDF5Replayer::openFLR(const char * hdf5_filename){
	if (imagereader_FLR != NULL)
		delete(imagereader_FLR);
	imagereader_FLR = new HDF5ImageReader(hdf5_filename);
	if (!imagereader_FLR)
		return false;
	totalframe_FLR = imagereader_FLR->n_images();
	curframe_FLR = 0;
	if (currentplaytime_usMax > 0 && currentplaytime_usMax < totalframe_FLR*samplingtime_FLRus) {
		currentplaytime_usMax = totalframe_FLR*samplingtime_FLRus;
		currentplaytime_us = 0;
	}
	curFLRimg.imgSize.height = imagereader_FLR->image_height();
	curFLRimg.imgSize.width = imagereader_FLR->image_width();
	if (curFLRimg.data) free(curFLRimg.data);
	curFLRimg.data = (uint16 *)malloc(curFLRimg.imgSize.height * curFLRimg.imgSize.width * sizeof(uint16));
	imagereader_NIR->read16(0, curFLRimg.data);
	return true;
}
bool HDF5Replayer::openDATA(const char * hdf5_filename){
	return true;
}

void HDF5Replayer::updateNIR_add(int frameDiff) {
	updateNIR(curframe_NIR + (INT64)frameDiff);
}
void HDF5Replayer::updateNIR(INT64 frame) {
	if (totalframe_NIR>0) {
		if (frame >= totalframe_NIR)
			frame = totalframe_NIR - 1;
		if (frame < 0)
			frame = 0;
		imagereader_NIR->read(frame, curNIRimg.data);
		curframe_NIR = frame;
	}
}
void HDF5Replayer::updateFLR_add(int frameDiff) {
	updateFLR(curframe_NIR + (INT64)frameDiff);
}
void HDF5Replayer::updateFLR(INT64 frame) {
	INT64 frame_load = frame + offsetframe_FLR;
	if (totalframe_FLR>0) {
		if (frame_load >= totalframe_FLR)
			frame_load = totalframe_FLR - 1;
		if (frame_load < 0)
			frame_load = 0;
		imagereader_FLR->read16(frame_load, curFLRimg.data);
		curframe_FLR = frame_load;
	}
}
void HDF5Replayer::update(void) {
	currentplaytime_us += updatingtime_us;
	if (currentplaytime_us > currentplaytime_usMax)
		currentplaytime_us = currentplaytime_usMax;
	if (currentplaytime_us < 0)
		currentplaytime_us = 0;
	updateNIR_time(currentplaytime_us);
	updateFLR_time(currentplaytime_us);
}
void HDF5Replayer::updateNIR_time(INT64 _time_us) {
	if (imagereader_NIR) {
		INT64 frm = _time_us / samplingtime_NIRus;
		updateNIR(frm);
	}
}
void HDF5Replayer::updateFLR_time(INT64 _time_us) {
	if (imagereader_FLR) {
		INT64 frm = _time_us / samplingtime_FLRus;
		updateFLR(frm);
	}
}

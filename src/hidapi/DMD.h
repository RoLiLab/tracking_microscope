#ifndef DMDHEADER_H
#define DMDHEADER_H

#include <windows.h>
#include "hidapi.h"

#define DMD_Reply 0xC0
#define DMD_buf_databegin 7
#define DMD_videomode 0x1A1B
#define DMD_max_imgloadsize 512
class DMD
{
public:
	DMD();
	~DMD();	
	hid_device *handle;
	unsigned char buff_cmd[65];
	unsigned char buff_ans[65];
	int res;
	int run;
	char devname[64]; // product name
	char errmsg[128]; // product name
	bool connect();
	bool disconnect();
	void cmd(bool reply, uint8 seq, uint16 cmd16, int datalen, uint8 * data = NULL);
	void start();
	void stop();
	void pause();
	//void hardware_status();
	enum{normalvideomode, prestored, videopattern, onthefly};
	void patterndisplayLUTdefinition(uint16 index, int exp_us, int dark_us, bool ext_trig, bool trigout2_enable, uint8 img_pattern_idx, uint8 bit_pos);
	void initializepatternBMPload(UINT32 imgsize_byte);
	void loadImage_half(bool master, byte * img, int imgsize_byte);
};

#endif // SERIALINTERFACEREADER_H

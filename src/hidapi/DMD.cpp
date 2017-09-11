#include "Base/base.h"
#include "DMD.h"

DMD::DMD(void)
{
	handle = NULL;
	memset(devname, 0, sizeof(devname));
	run = 0;
}

DMD::~DMD()
{
	disconnect();
}

bool DMD::connect(void) {
	if (handle)
		disconnect();
	handle = hid_open(0x0451, 0xc900, NULL);
	if (handle) {
		// Read the Product String
		wchar_t wstr[255];
		wstr[0] = 0x0000;
		res = hid_get_product_string(handle, wstr, 255);
		sprintf(devname, "%ls", wstr);
		return true;
	}
	return false;
}


bool DMD::disconnect(void) {
	if (handle) {
		stop();
		hid_close(handle);
		hid_exit();
		handle = NULL;
		run = 0;
		return true;
	}
	return false;
}

void DMD::cmd(bool reply, uint8 seq, uint16 cmd16, int datalen_byte, uint8 * data) {
	if (handle == NULL) return;
	memset(buff_cmd, 0, 6);
	buff_cmd[0] = 0x00;
	if (reply) {
		buff_cmd[1] = 0xc0; // w/r, reply required (1100 0000)
		memset(buff_ans, 0, sizeof(buff_ans));
	}
	buff_cmd[2] = seq;
	uint16 len_byte = uint16(datalen_byte + 2);
	*(uint16*)(&buff_cmd[3]) = len_byte;
	*(uint16*)(&buff_cmd[5]) = cmd16;
	if (datalen_byte > 0) {
		if (data != NULL)
			memcpy(&buff_cmd[DMD_buf_databegin], data, datalen_byte);
	}

	res = hid_write(handle, buff_cmd, 65);
	if (res < 0) {
		memset(errmsg, 0, sizeof(errmsg));
		sprintf(errmsg, "Error: %ls\n", hid_error(handle));
	}
	if (reply)
		res = hid_read(handle, buff_ans, sizeof(buff_ans));
}

void DMD::start(void) {
	if (handle == NULL) return;
	buff_cmd[DMD_buf_databegin] = 0x02; // start
	cmd(false, 0x00, 0x1A24, 1);
	run = 2;
}
void DMD::pause(void) {
	if (handle == NULL) return;
	buff_cmd[DMD_buf_databegin] = 0x01; // pause
	cmd(false, 0x00, 0x1A24, 1);
	run = 1;
}
void DMD::stop(void) {
	if (handle == NULL) return;
	buff_cmd[DMD_buf_databegin] = 0x00; // stop
	cmd(false, 0x00, 0x1A24, 1);
	run = 0;
}

void DMD::patterndisplayLUTdefinition(uint16 index, int exp_us, int dark_us, bool ext_trig, bool trigout2_enable, uint8 img_pattern_idx, uint8 bit_pos) {
	if (handle == NULL) return;
	// ----- set index
	if (index > 512) index = 511; // range 0-511
	*(uint16*)(&buff_cmd[DMD_buf_databegin]) = index;
	// -----  exposure time set (us)
	*(UINT32*)(&buff_cmd[DMD_buf_databegin + 2]) = UINT32(exp_us) << 8;
	// ----- pattern config and external trigger setting
	buff_cmd[DMD_buf_databegin + 5] = 7 << 4; // clear the pattern after exposure(false), bit depth (000<<2) : 1bit, LED enable(111<<4) : white
	if (ext_trig)
		buff_cmd[DMD_buf_databegin + 5] = buff_cmd[DMD_buf_databegin + 5] || 1 << 7; // wait for the external trigger
	// ----- dark time set (us)
	*(UINT32*)(&buff_cmd[DMD_buf_databegin + 6]) = UINT32(dark_us) << 8;
	// trigger output 2 setting
	if (trigout2_enable)
		buff_cmd[DMD_buf_databegin + 9] = 0x01; // enable external trigger 2
	else
		buff_cmd[DMD_buf_databegin + 9] = 0x00; // disable external trigger 2
	buff_cmd[DMD_buf_databegin + 10] = img_pattern_idx;
	buff_cmd[DMD_buf_databegin + 11] = bit_pos << 3;

	cmd(false, 0x00, 0x1A34, 11);
}

void DMD::initializepatternBMPload(UINT32 imgsize_byte) {
	if (handle == NULL) return;
	buff_cmd[DMD_buf_databegin] = 0x01;
	buff_cmd[DMD_buf_databegin + 1] = 0x00;
	*(UINT32*)(&buff_cmd[DMD_buf_databegin + 2]) = UINT32(imgsize_byte / 2) + 48;

	cmd(false, 0x00, 0x1A2A, 6);
	cmd(false, 0x00, 0x1A2C, 6);
}

void DMD::loadImage_half(bool master, byte * img, int imgsize_byte) {
	if (handle == NULL) return;

	unsigned char buf[518];
	buf[0] = 0x00;
	buf[1] = 0x00; // no reply
	buf[2] = 0x00; // seq
	uint16 len_byte = uint16(514);
	*(uint16*)(&buf[3]) = len_byte;
	if (master)
		*(uint16*)(&buf[5]) = 0x1A2B;
	else
		*(uint16*)(&buf[5]) = 0x1A2D;

	byte * curimg = img;
	int byte_transfered = 0;
	while (byte_transfered < imgsize_byte) {
		int byte_curtransfer = imgsize_byte - byte_transfered;
		if (byte_curtransfer > DMD_max_imgloadsize)
			byte_curtransfer = DMD_max_imgloadsize;
		memcpy(&buf[DMD_buf_databegin], curimg, byte_curtransfer);
		res = hid_write(handle, buf, byte_curtransfer + DMD_buf_databegin);
		byte_transfered += byte_curtransfer;
		curimg += byte_curtransfer;
	}
}

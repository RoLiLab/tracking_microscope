#ifndef XPSSERIALINTERFACEREADER_H
#define XPSSERIALINTERFACEREADER_H
//Direction | Name | Command byte | Data | Terminator | Notes
//---------- | ------ | -------------- | ------ | ------------ | ------ -
//H->D | On demand | 0x01 | XXYY | 0x0A |
//H->D | Reset | 0x02 | -| 0x0A |
//H->D | Start | 0x03 | -| 0x0A |
//H->D | Stop | 0x04 | -| 0x0A |
//H->D | Write | 0x05 | FXXYY | 0x0A |
//D->H | Wrote | 0x06 | FX | 0x0A | Fail: X = 0, Success : X = 1
//H->D | Set default | 0x07 | XXYY | 0x0A |

#include "SerialCOM.h"
#include <windows.h>
#include <stdint.h>

const uint8_t FLASH_CMD_OnDemand = 0x01;
const uint8_t FLASH_CMD_Reset = 0x02;
const uint8_t FLASH_CMD_Start = 0x03;
const uint8_t FLASH_CMD_Stop = 0x04;
const uint8_t FLASH_CMD_Write = 0x05;
const uint8_t FLASH_CMD_Wrote = 0x06;
const uint8_t FLASH_CMD_SetDefault = 0x07;
const uint8_t FLASH_CMD_Terminator = 0x0A;

struct TeensyMessageHandshake {
	TeensyMessageHandshake() : FrmNo(0), b_write(false), b_lateinput(false), MPCFrmIdxW(0), MPCFrmIdxR(0), MPCFrmIdxOffset(0) { CntrInput[0] = 0; CntrInput[1] = 0; MPCwrittenVoltage[0] = 0; MPCwrittenVoltage[1] = 0; }
	// input (from MPC controller -> Teensy)
	unsigned int FrmNo;
	double CntrInput[2];
	volatile bool b_write;
	volatile bool b_lateinput;
	uint8_t MPCFrmIdxW;
	uint8_t MPCFrmIdxR;
	uint8_t MPCFrmIdxOffset;
	double MPCwrittenVoltage[2];
	uint16_t MPCwrittenVoltageRawX;
	uint16_t MPCwrittenVoltageRawY;
};

class XPS_SerialCOM
{
public:
    XPS_SerialCOM();
    ~XPS_SerialCOM();
	SerialCOM handler;

	//------------- NewDesign2015 -------------- //
	volatile bool b_start;
	volatile bool b_stop;
	TeensyMessageHandshake msg_hsndshake;
	//------------------------------------------ //

	// XPS function
	void reset(void);
	void start(void);
	void stop(void);
	void sendVolInput(int Frame, double volX, double volY);
	bool readMsg(void);
	bool TrackingEnabled;
	void SetDefault(int16_t X, int16_t Y);
private:
	void Ondemand(int16_t X, int16_t Y);
	void writeVolMPC(uint8_t frm, int16_t X, int16_t Y);
};

#endif // SERIALINTERFACEREADER_H

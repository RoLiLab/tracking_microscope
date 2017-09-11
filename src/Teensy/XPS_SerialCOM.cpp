#include "Base/base.h"
#include "XPS_SerialCOM.h"

#include <windows.h>

XPS_SerialCOM::XPS_SerialCOM(void)
{
	TrackingEnabled = false;
	b_stop = false;
	b_start = false;
}

XPS_SerialCOM::~XPS_SerialCOM()
{
}

void XPS_SerialCOM::reset(void) {
	//PurgeComm(handler.hFile, PURGE_TXCLEAR | PURGE_RXCLEAR);
	handler.DataBuffer[0] = FLASH_CMD_Reset;
	handler.DataBuffer[1] = FLASH_CMD_Terminator;
	handler.write(handler.DataBuffer, 2);
}
void XPS_SerialCOM::start(void){
	PurgeComm(handler.hFile, PURGE_TXCLEAR | PURGE_RXCLEAR);
	handler.DataBuffer[0] = FLASH_CMD_Start;
	handler.DataBuffer[1] = FLASH_CMD_Terminator;
	handler.write(handler.DataBuffer, 2);
	TrackingEnabled = true;
}
void XPS_SerialCOM::stop(void){
	handler.DataBuffer[0] = FLASH_CMD_Stop;
	handler.DataBuffer[1] = FLASH_CMD_Terminator;
	handler.write(handler.DataBuffer, 2);
	TrackingEnabled = false;
}
void XPS_SerialCOM::writeVolMPC(uint8_t frm, int16_t X, int16_t Y){
	handler.DataBuffer[0] = FLASH_CMD_Write;
	handler.DataBuffer[1] = frm;
	*(uint16_t *)&handler.DataBuffer[2] = X;
	*(uint16_t *)&handler.DataBuffer[4] = Y;
	handler.DataBuffer[6] = FLASH_CMD_Terminator;
	handler.write(handler.DataBuffer, 7);
}
void XPS_SerialCOM::Ondemand(int16_t X, int16_t Y){
	handler.DataBuffer[0] = FLASH_CMD_OnDemand;
	*(uint16_t *)&handler.DataBuffer[1] = X;
	*(uint16_t *)&handler.DataBuffer[3] = Y;
	handler.DataBuffer[5] = FLASH_CMD_Terminator;
	handler.write(handler.DataBuffer, 6);
}

void XPS_SerialCOM::SetDefault(int16_t volX, int16_t volY){

	if (volX > 10) volX = 10;
	if (volX < -10) volX = -10;
	if (volY > 10) volY = 10;
	if (volY < -10) volY = -10;
	uint16_t X = UINT16_MAX*(volX + 10.0)/20.0;
	uint16_t Y = UINT16_MAX*(volY + 10.0)/20.0;

	handler.DataBuffer[0] = FLASH_CMD_SetDefault;
	*(uint16_t *)&handler.DataBuffer[1] = X;
	*(uint16_t *)&handler.DataBuffer[3] = Y;
	handler.DataBuffer[5] = FLASH_CMD_Terminator;
	handler.write(handler.DataBuffer, 6);
}


void XPS_SerialCOM::sendVolInput(int Frame, double volX, double volY) {
	if (volX > 10) volX = 10;
	if (volX < -10) volX = -10;
	if (volY > 10) volY = 10;
	if (volY < -10) volY = -10;
	uint16_t X = 65535*(volX + 10.0)/20.0;
	uint16_t Y = 65535*(volY + 10.0)/20.0;

	if (Frame == 0) { // ondemand
		Ondemand(X, Y);
	}
	else {// MPC
		msg_hsndshake.MPCFrmIdxW = (uint8_t)(Frame % (256));
		writeVolMPC(msg_hsndshake.MPCFrmIdxW, X, Y);
	}
}



bool XPS_SerialCOM::readMsg(void) {
	int Bytes_expectedRead = 4;
	handler.read(Bytes_expectedRead);
	if (handler.DataBuffer[0] == FLASH_CMD_Wrote) {
		msg_hsndshake.MPCFrmIdxR = ((uint8_t)handler.DataBuffer[1]);
		msg_hsndshake.MPCFrmIdxOffset = msg_hsndshake.MPCFrmIdxW - msg_hsndshake.MPCFrmIdxR;
		//msg_hsndshake.MPCwrittenVoltageRawX = *(uint16_t *)&handler.DataBuffer[2];
		//msg_hsndshake.MPCwrittenVoltageRawY = *(uint16_t *)&handler.DataBuffer[4];
		//msg_hsndshake.MPCwrittenVoltage[0] = ((double)msg_hsndshake.MPCwrittenVoltageRawX - 32767) / 65535 * 10;
		//msg_hsndshake.MPCwrittenVoltage[1] = ((double)msg_hsndshake.MPCwrittenVoltageRawY - 32767) / 65535 * 10;
		//if (msg_hsndshake.MPCwrittenVoltageRawX > 0 && msg_hsndshake.MPCwrittenVoltageRawY > 0) {
		if (((uint8_t)handler.DataBuffer[2]) == 1) {
			msg_hsndshake.b_lateinput = false;
		}
		else {
			msg_hsndshake.b_lateinput = true;
		}
	}
	return msg_hsndshake.b_lateinput;
}

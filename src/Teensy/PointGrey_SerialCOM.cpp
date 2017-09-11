#include "Base/base.h"
#include "PointGrey_SerialCOM.h"

#include <windows.h>

POINTGREY_SerialCOM::POINTGREY_SerialCOM(void)
{
	ResolutionUS_LSB = 4;
}

POINTGREY_SerialCOM::~POINTGREY_SerialCOM()
{
	handler.close();
}


void POINTGREY_SerialCOM::POINTGREY_setExptimeUS(unsigned int exposure_us) {
	unsigned int n = exposure_us / ResolutionUS_LSB;
	if (n > UINT8_MAX) n = UINT8_MAX;
	handler.DataBuffer[0]= (uint8_t)n;
	handler.write(handler.DataBuffer, 1);
}

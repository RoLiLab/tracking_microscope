#ifndef POINTGREYSERIALINTERFACEREADER_H
#define POINTGREYSERIALINTERFACEREADER_H

#include "SerialCOM.h"
#include <windows.h>


class POINTGREY_SerialCOM
{
public:
    POINTGREY_SerialCOM();
    ~POINTGREY_SerialCOM();
	SerialCOM handler;
	
	// Camera function
	void POINTGREY_setExptimeUS(unsigned int dt);
	unsigned int ResolutionUS_LSB;
};

#endif // SERIALINTERFACEREADER_H

#ifndef SERIALINTERFACEREADER_H
#define SERIALINTERFACEREADER_H

#include <windows.h>

class SerialCOM
{
public:
    SerialCOM();
    ~SerialCOM();
	HANDLE hFile; 
	bool bErrorFlag;
	uint8_t DataBuffer[256];
	DWORD dwBytesWritten;
	DWORD bytes_read;
	BOOL open(char * IpFileName);
	BOOL close(void);
	void read(int Bytes_expectedRead);
	BOOL write(void * DataBuffer, DWORD nNumberOfBytesToWrite);
	BOOL clearMsg(void);
};

#endif // SERIALINTERFACEREADER_H

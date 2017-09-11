#include "Base/base.h"
#include "SerialCOM.h"

SerialCOM::SerialCOM(void)
{
	hFile = INVALID_HANDLE_VALUE;
    bErrorFlag = FALSE;
	dwBytesWritten = 0;
	bytes_read = 0;
}

SerialCOM::~SerialCOM()
{
	close();
}

BOOL SerialCOM::open(char * FileName){
	const size_t cSize = strlen(FileName)+1;
    wchar_t* IpFileName = new wchar_t[cSize];
    mbstowcs (IpFileName, FileName, cSize);
	close();

	//CreateFileW uses a UTF16 unicode string for the filename.
	hFile = CreateFileW(IpFileName, // "\\\\.\\com36",// name of the write
                       GENERIC_READ | GENERIC_WRITE,			   // open ALL
                       0,                      // do not share
                       NULL,                   // default security
                       OPEN_EXISTING,             // create new file only
                       FILE_ATTRIBUTE_NORMAL,  // normal file
                       NULL);                  // no attr. template

	if (hFile == INVALID_HANDLE_VALUE)
		return false;

	PurgeComm(hFile, PURGE_TXCLEAR | PURGE_RXCLEAR);
	return true;
}

BOOL SerialCOM::clearMsg(void) {
	if (hFile == INVALID_HANDLE_VALUE)
		return false;
	PurgeComm(hFile, PURGE_TXCLEAR | PURGE_RXCLEAR);
	Sleep(4000);
	return true;
}


BOOL SerialCOM::close(void){
	if (hFile == INVALID_HANDLE_VALUE)
		return false;
    CloseHandle(hFile);
	hFile = INVALID_HANDLE_VALUE;
	return true;
}

BOOL SerialCOM::write(void * DataBuffer, DWORD nNumberOfBytesToWrite){
	if (hFile == INVALID_HANDLE_VALUE) return false;
    bErrorFlag = WriteFile(
                    hFile,           // open file handle
                    DataBuffer,       // start of data to write
                    nNumberOfBytesToWrite,  // number of bytes to write
                    &dwBytesWritten, // number of bytes that were written
                    NULL);            // no overlapped structure
	return bErrorFlag;
}

void SerialCOM::read(int Bytes_expectedRead){
	if (hFile == INVALID_HANDLE_VALUE) return;

	DWORD  bytes_left = (DWORD)Bytes_expectedRead; //bytes
	bytes_read = 0;
	uint8_t * p_read = DataBuffer;
	while (bytes_left > 0) {
		bErrorFlag == ReadFile(
			hFile,
			p_read,
			bytes_left,
			&bytes_read,
			NULL);
		p_read += bytes_read;
		bytes_left -= bytes_read;
	}
}

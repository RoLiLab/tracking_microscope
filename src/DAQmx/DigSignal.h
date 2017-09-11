#ifndef _DigSignal_H
#define _DigSignal_H

#include "NIDAQmx.h"

class DigSignal
{
public:
	// basic functions
	DigSignal(void);
	~DigSignal(void);
	DigSignal(int edgeCount, uInt64 * edgeNo, uInt32 initSignal); // to generate sample signal
	DigSignal(uInt64 _n, uInt32 initSignal); // to generate general signal

	uInt32 * data;
	uInt64 n;

	void resetSignal(uInt64 _n, uInt32 initSignal);
	void generateDigSignal(int edgeCount, uInt64 * edgeNo, uInt32 initSignal);
	void mergeSignal(DigSignal * SigA, uInt64 offset, uInt64 startCountNo, uInt32 initValue);
	void deepcopy(DigSignal * SigA);
private:
	uInt64 getSampleIdxFromGlobalIdx(uInt64 globalIdx, uInt64 signallength, uInt64 offset);
};

void orUINT32(uInt64 length, uInt32 * org, uInt32 * cpy) ;
void orUINT32_single(uInt64 length, uInt32 * org, uInt32 value);
#endif

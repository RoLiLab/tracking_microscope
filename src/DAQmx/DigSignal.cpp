#include "Base/base.h"
#include "DAQmx/DigSignal.h"

DigSignal::DigSignal(void)
{
	data = NULL;
	n = 0;
}

DigSignal::~DigSignal(void)
{
	if (data) free(data);
}

DigSignal::DigSignal(int edgeCount, uInt64 * edgeNo, uInt32 initSignal)
{
	data = NULL;
	generateDigSignal(edgeCount, edgeNo, initSignal);
}

DigSignal::DigSignal(uInt64 _n, uInt32 initSignal)
{
	data = NULL;
	resetSignal(_n, initSignal);
}

void DigSignal::resetSignal(uInt64 _n, uInt32 initSignal)
{
	if (data) free(data);
	n = _n;
	data = (uInt32 *) malloc (n* sizeof(uInt32));
	for (int i = 0; i < n; i++) {
		data[i] = initSignal;
	}
}

void DigSignal::generateDigSignal(int edgeCount, uInt64 * edgeNo, uInt32 initSignal)
{
	// when edgeNo[0] = 0, the sequence of the signal will be 0x00 0x** 0x00 0x** ...
	// when edgeNo[0] = k, the sequence of the signal will be 0x** 0x00 ...
	if (data) free(data);
	n = edgeNo[edgeCount - 1];
	data = (uInt32 *) malloc (n* sizeof(uInt32));
	uInt32 Signal[2] = {initSignal, 0};

	int EdgeCount_Cur = 0;
	int Signal_index = 0;
	for (int i = 0; i < n; i++) {
		//data[i] = 0; // initialization
		if (edgeNo[EdgeCount_Cur] == i) {
			Signal_index = (++EdgeCount_Cur)%2;
		}
		data[i] = Signal[Signal_index];
	}
}

void DigSignal::mergeSignal(DigSignal * SigA, uInt64 offset, uInt64 startCountNo, uInt32 initValue) {

	uInt64 globalIdx;
	uInt64 copyLength;

	if (startCountNo < offset) {
		orUINT32_single(offset, data, initValue);
		globalIdx = offset;
	}
	else {
		globalIdx = startCountNo;
		uInt64 Idx = getSampleIdxFromGlobalIdx(startCountNo, SigA->n, offset);
		copyLength = SigA->n - Idx;
		orUINT32(copyLength, data, SigA->data + Idx);
		globalIdx = globalIdx + copyLength;
	}

	// from globalIdx to (globalIdx + copyLength)
	while (globalIdx < startCountNo+n) {
		uInt64 nextglobalIdx = globalIdx + SigA->n;
		if (nextglobalIdx > startCountNo+n) {
			copyLength = getSampleIdxFromGlobalIdx(startCountNo+n, SigA->n, offset);
		}
		else {
			copyLength = SigA->n;
		}
		orUINT32(copyLength, data+globalIdx-startCountNo, SigA->data);
		globalIdx = nextglobalIdx;
	}
}
uInt64 DigSignal::getSampleIdxFromGlobalIdx(uInt64 globalIdx, uInt64 signallength, uInt64 offset) {
	// global index = offset + (signal length) * share + (local index)
	uInt64 share = (globalIdx - offset)/signallength;
	return globalIdx - (signallength * share + offset);
}

void DigSignal::deepcopy(DigSignal * SigA) {
	resetSignal(SigA->n, 0.0);
	memcpy(data, SigA->data, sizeof(uInt32)*SigA->n);
}

void orUINT32(uInt64 length, uInt32 * org, uInt32 * cpy) {
	for (uInt64 i = 0; i < length; i++)
		org[i] |= cpy[i];
}

void orUINT32_single(uInt64 length, uInt32 * org, uInt32 value) {
	for (uInt64 i = 0; i < length; i++)
		org[i] |= value;
}

#include "Base/base.h"
#include "DAQmx/AnalogSignal.h"
#include "NIDAQmx.h"


AnalogSignal::AnalogSignal(void)
{
	AO_type = AO_Sawtooth;
	data = NULL;
	n = 0;
}

AnalogSignal::~AnalogSignal(void)
{
	if (data) free(data);
}

AnalogSignal::AnalogSignal(int _AO_type, uInt64 _n, float64 p2, float64 p3) // to generate sample signal
{
	data = NULL;
	generateAnalogSignal(_AO_type, _n, p2, p3);
}

AnalogSignal::AnalogSignal(uInt64 _n, float64 initSignal)
{
	data = NULL;
	resetSignal(_n, initSignal);
}

void AnalogSignal::resetSignal(uInt64 _n, float64 initSignal)
{
	if (data) free(data);
	n = _n;
	data = (float64 *) malloc (n* sizeof(float64));
	for (int i = 0; i < n; i++) {
		data[i] = initSignal;
	}
}

void AnalogSignal::resetSignal_value(float64 initSignal)
{
	generateCONSTANT(initSignal);
}

void AnalogSignal::generateAnalogSignal(int _AO_type, uInt64 _n, float64 p2, float64 p3)
{
	// _AO_type -> enum{AO_Sawtooth, AO_Triangle, AO_Sine, AO_Constant};
	// default 1us resolution (p1 is frequency -> n (us) = 1/f*1000*1000
	// p2 is minimum value, p3 is maximum value
	AO_type = _AO_type;
	float64 * data_new;
	data_new = (float64 *) malloc (2*_n* sizeof(float64));
	switch(AO_type) {
		case AO_Sawtooth: //
			generateSAWTOOTH(p2, p3, data_new, _n);
			break;
		case AO_Triangle: //
			generateTRIANGLE(p2, p3);
			break;
		case AO_Sine: //
			generateSINE(p2, p3);
			break;
		case AO_Square: //
			generateSQUARE(p2, p3);
			break;
		case AO_Constant: //
			generateCONSTANT(p2);
			break;
		}; // end switch
	if (n > 2*_n) {
		n = 2*_n;
		float64 * temp = data;
		data = data_new;
		if (temp) free(temp);
	}
	else {
		float64 * temp = data;
		data = data_new;
		if (temp) free(temp);
		n = 2*_n;
	}
}

void AnalogSignal::generateSAWTOOTH(float64 _min, float64 _max, float64 * _data, int _n )
{
	// (_max - _min) / (n-1) * i + _min
	float64 d = (_max - _min) / (float64)(_n-1);
	for (int i = 0 ; i < _n; i++)
		_data[i] = d * (float64)i + _min;
	for (int i = _n ; i < 2*_n; i++)
		_data[i] = _data[(2*_n-i)-1];
}
void AnalogSignal::generateTRIANGLE(float64 _min, float64 _max)
{
}
void AnalogSignal::generateSINE(float64 _min, float64 _max)
{
}
void AnalogSignal::generateSQUARE(float64 _min, float64 _max)
{
}
void AnalogSignal::generateCONSTANT(float64 _v)
{
	for (int i = 0; i < n; i++) {
		data[i] = _v;
	}
}
uInt64 AnalogSignal::getSampleIdxFromGlobalIdx(uInt64 globalIdx, uInt64 signallength, uInt64 offset) {
	// global index = offset + (signal length) * share + (local index)
	uInt64 share = (globalIdx - offset)/signallength;
	return globalIdx - (signallength * share + offset);
}

void AnalogSignal::generateAnalogSignal(AnalogSignal * SigA, uInt64 offset, uInt64 startCountNo, uInt32 initValue) {

	uInt64 globalIdx;
	uInt64 copyLength;

	if (startCountNo < offset) {
		for (int i = 0; i < offset; i++)
			data[i] = initValue;
		globalIdx = offset;
	}
	else {
		globalIdx = startCountNo;
		uInt64 Idx = getSampleIdxFromGlobalIdx(startCountNo, SigA->n, offset);
		copyLength = SigA->n - Idx;
		if (copyLength > n)
			copyLength = n;
		memcpy(data, SigA->data + Idx, copyLength*sizeof(float64));
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
		memcpy(data+globalIdx-startCountNo, SigA->data, copyLength*sizeof(float64));
		globalIdx = nextglobalIdx;
	}
}

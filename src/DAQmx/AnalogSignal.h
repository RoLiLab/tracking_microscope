#ifndef _AnalogSignal_H
#define _AnalogSignal_H
#include "NIDAQmx.h"

class AnalogSignal
{
public:

	// basic functions
	AnalogSignal(void);
	~AnalogSignal(void);
	AnalogSignal(int _AO_type, uInt64 _n, float64 p2, float64 p3); // to generate sample signal
	AnalogSignal(uInt64 _n, float64 initSignal); // to generate general signal

	float64 * data;
	uInt64 n;
	int AO_type;

	void resetSignal(uInt64 _n, float64 initSignal);
	void resetSignal_value(float64 initSignal);
	void generateAnalogSignal(int _AO_type, uInt64 n, float64 p2, float64 p3);
	void generateAnalogSignal(AnalogSignal * SigA, uInt64 offset, uInt64 startCountNo, uInt32 initValue);

	enum{AO_Sawtooth, AO_Triangle, AO_Sine, AO_Square, AO_Constant};
private:
	uInt64 getSampleIdxFromGlobalIdx(uInt64 globalIdx, uInt64 signallength, uInt64 offset);
	void generateSAWTOOTH(float64 _min, float64 _max, float64 * _data, int _n);
	void generateTRIANGLE(float64 _min, float64 _max);
	void generateSINE(float64 _min, float64 _max);
	void generateSQUARE(float64 _min, float64 _max);
	void generateCONSTANT(float64 _v);
};

#endif

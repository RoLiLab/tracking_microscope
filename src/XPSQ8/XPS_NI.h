#ifndef STAGENI_H
#define STAGENI_H

#include "DAQmx/DigSignal.h"

class XPS_NI
{
public:
    XPS_NI();
	XPS_NI(uInt64 _cycle_us);
    ~XPS_NI();
	
	uInt64 cycle_us;
	uInt64 cam_offset;
	DigSignal signal_CamTrigger;
	DigSignal signal_Stage;
	DigSignal signal_Toggle;
	DigSignal signal;
	void getSignal(uInt64 _cycle_us);

private:
};

#endif // SERIALINTERFACEREADER_H

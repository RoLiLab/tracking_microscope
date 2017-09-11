#include "Base/base.h"
#include "XPS_NI.h"

XPS_NI::XPS_NI(void)
{
	cam_offset = 62;
	getSignal(4000);
}
XPS_NI::XPS_NI(uInt64 _cycle_us)
{
	getSignal(_cycle_us);
}

XPS_NI::~XPS_NI()
{
}


void XPS_NI::getSignal(uInt64 _cycle_us) {
	cycle_us = _cycle_us;
	
	uInt64 edgeNo[3] = {0};
	edgeNo[0] = cycle_us/2;
	edgeNo[1] = cycle_us;
	signal_Stage.generateDigSignal(2, edgeNo, 1);

	edgeNo[0] = cycle_us;
	edgeNo[1] = cycle_us * 2;
	signal_Toggle.generateDigSignal(2, edgeNo, 1<<1);

	edgeNo[0] = cycle_us / 2;
	edgeNo[1] = cycle_us;
	signal_CamTrigger.generateDigSignal(2, edgeNo, 1 << 5);

	signal.resetSignal(2*cycle_us, 0);
	signal.mergeSignal(&signal_Stage, 0, 0, 0);
	signal.mergeSignal(&signal_Toggle, 0, 0, 0);
}
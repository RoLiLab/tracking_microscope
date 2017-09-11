#ifndef LIBUSBDEVICEDESCRIPTOR_H
#define LIBUSBDEVICEDESCRIPTOR_H

#include "libusb.h"

const uint8_t k_pulse_duration0 = 0x00;
const uint8_t k_pulse_duration1 = 0x01;
const uint8_t k_min_trigger_interval = 0x02;
const uint8_t k_pulse_index = 0x03;
const uint8_t k_timebase_frequency = 0x04;

class Libusb_device_descriptor
{
public:
	Libusb_device_descriptor();
	~Libusb_device_descriptor();
	
	bool find_device(uint16, uint16);
	uint8 get_register(uint8);
	bool set_register(uint8, uint8);

	int error;
	libusb_device_handle * handler;
};

#endif // SERIALINTERFACEREADER_H

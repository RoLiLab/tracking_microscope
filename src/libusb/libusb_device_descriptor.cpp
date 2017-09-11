#include "Base/base.h"
#include "libusb_device_descriptor.h"

Libusb_device_descriptor::Libusb_device_descriptor(void)
{
	error = 0;
	handler = NULL;
}

Libusb_device_descriptor::~Libusb_device_descriptor()
{
	if (handler)
		libusb_close(handler);
	libusb_exit(NULL);
}

bool Libusb_device_descriptor::find_device(uint16 id_vendor, uint16 id_product) {

	libusb_device * dev;
	libusb_device **devs;
	ssize_t cnt;

	error = libusb_init(NULL);
	if (error != LIBUSB_SUCCESS)
		return false;

	cnt = libusb_get_device_list(NULL, &devs);
	if (cnt < 0)
		return false;

	libusb_device_descriptor * dev_descriptor;


	for (int i = 0; i < cnt; i++) {
		libusb_device * cur_dev = devs[i];
		error = libusb_get_device_descriptor(cur_dev, dev_descriptor);
		if (error == LIBUSB_SUCCESS) {
			if (dev_descriptor->idVendor == id_vendor && dev_descriptor->idVendor == id_product) {
				dev = cur_dev;
				break;
			}
		}
	}
	if (dev == NULL)
		return false;

	if (handler)
		libusb_close(handler);

	error = libusb_open(dev, &handler);
	if (error != LIBUSB_SUCCESS) {
		handler = NULL;
		return false;
	}

	return true;
}

uint8 Libusb_device_descriptor::get_register(uint8 reg) {
	uint16 index = 0;
	uint8 data = 0;
	error = libusb_control_transfer(handler, 0xc0, reg, 0,
		index, &data, 4, 1000);
	if (error != LIBUSB_SUCCESS)
		return 0;
	return data;
}
bool Libusb_device_descriptor::set_register(uint8 reg, uint8 value) {
	uint16 index = 0;
	error = libusb_control_transfer(handler, 0x40, reg, 0,
		index, &value, 4, 1000);
	if (error != LIBUSB_SUCCESS)
		return false;
	return true;
}

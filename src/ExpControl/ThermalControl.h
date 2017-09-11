#ifndef THERMALCONTROL_H
#define THERMALCONTROL_H

#include <windows.h>
#include "Base/base.h"
class ThermalControl
{
public:
	ThermalControl();
	~ThermalControl();
	void updateparameters(void);
	double getinput(double * loc, int _frm);
	double getinput_time(int _frm);
	double getinput_map(double * loc);
	double getinput_cross(double * loc);
	double getinput_half(double * loc);
	double getinput_half_repeat(double * loc, int _frm);
	double getinput_halfinv(double * loc);	
	double getinput_halfgrad(double * loc);
	double getinput_halfgradinv(double * loc);
	double getinput_halfgrad_repeat(double * loc, int _frm);
	double getinput_circle(double * loc);
	double getinput_circle_rev(double * loc, int _frm);
	
	
	int mode;
	bool enable;
	double voltage_max;
	double voltage_min;
	double voltage;
	double dx_mm;
	double dy_mm;
	double cx_mm;
	double cy_mm;
	int cx_px;
	int cy_px;
	double cycle_sec;
	double on_sec;
	int frm_rate;
	int frm_on;
	int frm_total;
	int imgsize_x;
	int imgsize_y;

	bool Osc_enable;
	double Osc_Amp;
	double Osc_Period;
	double * Osc_data;
	int Osc_data_n;
	int Osc_data_n_cur;

	uint64 circle_inv_counter;
	uint64 circle_inv_counter_cooldown;
	uint64 circle_inv_counter_max;
	double circle_inv_rmin;
	bool circle_inv_enable;
	bool circle_inv_cooldown_enable;

	uint8 * map;
};

#endif // THERMALCONTROL_H

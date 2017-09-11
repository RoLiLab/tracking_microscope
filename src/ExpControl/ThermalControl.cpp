#include "Base/base.h"
#include "ThermalControl.h"

ThermalControl::ThermalControl(void)
{
	mode = 0;
	enable = false;
	voltage_max = 10;
	voltage_min = 0;
	voltage = 0;
	dx_mm = 0.1;
	dy_mm = 0.1;
	cx_px = 0;
	cy_px = 0;
	cx_mm = 0;
	cy_mm = 0;
	cycle_sec = 10.0;
	on_sec = 1.0;
	frm_rate = 250;
	frm_total = (int)((double)frm_rate*cycle_sec);
	frm_on = (int)((double)frm_rate*on_sec);
	imgsize_x = 0;
	imgsize_y = 0;

	Osc_Amp = 0;
	Osc_Period = 0;
	Osc_data_n = 0;
	Osc_data_n_cur = 0;
	Osc_data = NULL;
	Osc_enable = false;

	circle_inv_counter = 0;
	circle_inv_counter_cooldown = 250 * 60 * 10;
	circle_inv_counter_max = 250*60*11;
	circle_inv_rmin = 10;
	circle_inv_enable = false;
	map = NULL;
}

ThermalControl::~ThermalControl()
{
}
void ThermalControl::updateparameters(void)
{
	if (!enable) voltage = 0;
	if (voltage_max > 10) voltage_max = 10;	else if (voltage_max < 0) voltage_max = 0;
	if (voltage_min > voltage_max) voltage_min = voltage_max;	else if (voltage_min < 0) voltage_min = 0;
	if (cycle_sec < on_sec) on_sec = cycle_sec;
	frm_total = (int)((double)frm_rate*cycle_sec);
	frm_on = (int)((double)frm_rate*on_sec);
	//if (cx_px > imgsize_x) cx_px = imgsize_x;	else if (cx_px < 0) cx_px = 0;
	//if (cy_px > imgsize_y) cy_px = imgsize_y;	else if (cy_px < 0) cy_px = 0;
	double dt = 0.004;
	int n = round(Osc_Period / dt);
	double omega = 2 * PI / Osc_Period;
	double * _data = (double *)malloc(sizeof(double) * n);
	for (int i = 0; i < n; i++)
		_data[i] = Osc_Amp * sin(omega * i * dt);

	double * temp = Osc_data;
	Osc_data_n_cur = 0;
	Osc_data = _data;
	Osc_data_n = n;

	circle_inv_counter = 0;

	if (temp) free(temp);
}


double ThermalControl::getinput(double * loc, int _frm) {
	if (enable) {
		switch (mode) {
		case 0:
			voltage = getinput_time(_frm);
			break;
		case 1:
			voltage = getinput_cross(loc);
			break;
		case 2:
			voltage = getinput_map(loc);
			break;
		case 3:
			voltage = getinput_half(loc);
			break;
		case 4:
			voltage = getinput_halfinv(loc);
			break;
		case 5:
			voltage = getinput_half_repeat(loc, _frm);
			break;
		case 6:
			voltage = getinput_halfgrad(loc);
			break;
		case 7:
			voltage = getinput_halfgradinv(loc);
			break;
		case 8:
			voltage = getinput_halfgrad_repeat(loc, _frm);
			break;
		case 9:
			voltage = getinput_circle(loc);
			break;
		case 10:
			voltage = getinput_circle_rev(loc, _frm);
			break;
		default:
			voltage = 0;
			break;
		}

		if (Osc_enable && Osc_data) {
			voltage += Osc_data[(Osc_data_n_cur++) % Osc_data_n];
		}
	}
	else
		voltage = 0;


	return voltage;
}
double ThermalControl::getinput_time(int _frm) {
	double vol = 0;
	int i = _frm%frm_total;
	if (i <= frm_on)
		vol = voltage_max;
	else
		vol = voltage_min;
	return vol;
}
double ThermalControl::getinput_map(double * loc) {
	double vol = 0;
	if (map){
		int x_px = int(loc[0] / dx_mm) + cx_px;
		int y_px = int(loc[1] / dy_mm) + cy_px;
		if (x_px < 0) x_px = 0; else if (x_px >= imgsize_x) x_px = imgsize_x - 1;
		if (y_px < 0) y_px = 0; else if (y_px >= imgsize_y) y_px = imgsize_y - 1;
		//int linearidx = x_px*imgsize_y + y_px;
		int linearidx = y_px*imgsize_x + x_px;
		vol = map[linearidx] / UINT8_MAX*(voltage_max - voltage_min) + voltage_min;
	}
	return vol;
}

double ThermalControl::getinput_cross(double * loc) {
	double vol = 0;
	if ((fabs(loc[0] - cx_mm) < dx_mm/2) || (fabs(loc[1] - cy_mm) < dy_mm/2)) {
		vol = voltage_min;
	}
	else
		vol = voltage_max;
	return vol;
}

double ThermalControl::getinput_half(double * loc) {
	double vol = 0;
	if ((loc[0] - cx_mm) < 0)
		vol = voltage_min;
	else
		vol = voltage_max;
	return vol;
}

double ThermalControl::getinput_half_repeat(double * loc, int _frm) {
	double vol = 0;
	double _x = (loc[0] - cx_mm);

	circle_inv_counter = _frm%circle_inv_counter_max;
	if (circle_inv_counter > circle_inv_counter_cooldown)
		circle_inv_cooldown_enable = true;
	else
		circle_inv_cooldown_enable = false;

	double dp = loc[0] - cx_mm;
	if (circle_inv_cooldown_enable)
		dp = -dp;

	if (dp < 0)
		vol = voltage_min;
	else
		vol = voltage_max;
	return vol;
}

double ThermalControl::getinput_halfinv(double * loc) {
	double vol = 0;
	if ((loc[0] - cx_mm) < 0)
		vol = voltage_max;
	else
		vol = voltage_min;
	return vol;
}

double ThermalControl::getinput_halfgrad(double * loc) {
	double vol = 0;
	if ((fabs(loc[0] - cx_mm) < dx_mm)) {
		vol = (voltage_max - voltage_min) * ((loc[0] - cx_mm) + dx_mm) / (2 * dx_mm) + voltage_min;
	}
	else {
		if (loc[0] - cx_mm < dx_mm)
			vol = 0;
		else
			vol = voltage_max;
	}
	return vol;
}

double ThermalControl::getinput_halfgradinv(double * loc) {
	double vol = 0;
	if ((fabs(loc[0] - cx_mm) < dx_mm)) {
		vol = (voltage_max - voltage_min) * (-((loc[0] - cx_mm) + dx_mm)) / (2 * dx_mm) + voltage_min;
	}
	else {
		if (loc[0] - cx_mm < dx_mm)
			vol = voltage_max;
		else
			vol = 0;
	}
	return vol;
}


double ThermalControl::getinput_halfgrad_repeat(double * loc, int _frm) {
	double vol = 0;
	double _x = (loc[0] - cx_mm);

	circle_inv_counter = _frm%circle_inv_counter_max;
	if (circle_inv_counter > circle_inv_counter_cooldown)
		circle_inv_cooldown_enable = true;
	else
		circle_inv_cooldown_enable = false;

	if (circle_inv_cooldown_enable)
		_x = -_x;

	if (fabs(_x) < dx_mm) {
		vol = ((voltage_max - voltage_min) * (_x + dx_mm) / (2 * dx_mm)) + voltage_min;
		//(dx_mm - -dx_mm):(voltage_max -voltage_min) = (_x - -dx_mm):(vol - voltage_min)
	}
	else {
		if (_x < dx_mm)
			vol = voltage_min;
		else
			vol = voltage_max;
	}
	return vol;
}

double ThermalControl::getinput_circle(double * loc) {
	double vol = 0;
	double _x = (loc[0] - cy_mm);
	double _y = (loc[1] - cy_mm);
	double r = sqrt(_x*_x + _y*_y);

	double minr = dx_mm;
	double maxr = dy_mm;
	double vol_min = voltage_min;
	double vol_max = voltage_max;

	if (r > minr) {
		if (r < maxr)
			vol = (vol_max - vol_min) * (r - minr) / (maxr - minr) + vol_min;
			// (voltage_max - voltage_min) : (dy_mm - dx_mm) = (vol - voltage_min) : (r - dx_mm)
		else
			vol = vol_max;
	}
	else {
		vol = vol_min;
	}
	return vol;
}


double ThermalControl::getinput_circle_rev(double * loc, int _frm) {
	double vol = 0;
	double _x = (loc[0] - cx_mm);
	double _y = (loc[1] - cy_mm);
	double r = sqrt(_x*_x + _y*_y);

	circle_inv_counter = _frm%circle_inv_counter_max;
	if (circle_inv_counter > circle_inv_counter_cooldown)
		circle_inv_cooldown_enable = true;
	else
		circle_inv_cooldown_enable = false;

	double minr = dx_mm;
	double maxr = dy_mm;
	double vol_min = voltage_min;
	double vol_max = voltage_max;
	if (circle_inv_cooldown_enable  && circle_inv_enable) {
		minr = circle_inv_rmin;
		vol_min = voltage_max;
		vol_max = voltage_min;
	}

	if (r > minr) {
		if (r < maxr)
			vol = (vol_max - vol_min) * (r - minr) / (maxr - minr) + vol_min;
		// (voltage_max - voltage_min) : (dy_mm - dx_mm) = (vol - voltage_min) : (r - dx_mm)
		else
			vol = vol_max;
	}
	else {
		vol = vol_min;
	}
	return vol;
}

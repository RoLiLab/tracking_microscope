#include "Base/base.h"
#include "XPS_I2T.h"

XPS_I2T::XPS_I2T(void)
{
	_I2T_Buffer = NULL;
	set(ScalingAcc_X, 0.004, 0.002, 0.05, 0.1, 0.2);
}

XPS_I2T::~XPS_I2T()
{
	if (_I2T_Buffer) 
		free(_I2T_Buffer);
}

void XPS_I2T::set(double _ScalingAcc, double dt, double L2w_low, double L2w_upper, double L2w_th_low, double L2w_th_upper) {
	AccelerationLimit = _ScalingAcc/1.05;
	_AccelerationLimit_max = _ScalingAcc / 1.05;
	_AccelerationLimitPeak = _ScalingAcc / 1.05 / I2TDriverMaximumCurrent * I2TDriverMaximumPeakCurrent;
	_AccelerationLimitRMS = _ScalingAcc / 1.05 / I2TDriverMaximumCurrent * I2TDriverMaximumRMSCurrent;;
	AccelerationLimit_normal = _AccelerationLimit_max;
	AccelerationLimit_low = _AccelerationLimitRMS;
	_I2TPeak = (I2TDriverMaximumPeakCurrent*I2TDriverMaximumPeakCurrent) * 4;
	_I2TRMS = (I2TDriverMaximumRMSCurrent*I2TDriverMaximumRMSCurrent) * I2TDriverRMSIntegrationTime * 0.9;
	_dt = dt;
	_I2T_BufferCountMax = (int)(I2TDriverRMSIntegrationTime / _dt);
	_I2T_BufferCount = 0;
	if (_I2T_Buffer)
		free(_I2T_Buffer);
	_I2T_Buffer = (double *)malloc(_I2T_BufferCountMax * sizeof(double));
	for (int i = 0; i < _I2T_BufferCountMax; i++)
		_I2T_Buffer[i] = 0;
	_den = (I2TDriverMaximumPeakCurrent * I2TDriverMaximumPeakCurrent * _dt) / (_AccelerationLimitPeak * _AccelerationLimitPeak);
	_sum = 0;
	_I2T = 0;
	_I2T_MAX = 0;
	currentlimitedmodeenabled = false;
	set_L2w(L2w_low, L2w_upper, L2w_th_low, L2w_th_upper);
	counter_recoverytime = 0;
	counter_totaltime = 0;
	counter_totaltime_max = 1000;
	dutycycle = 1.0;
}
void XPS_I2T::set_L2w(double L2w_low, double L2w_upper, double L2w_th_low, double L2w_th_upper) {
	L2w[0] = L2w_low; 
	L2w[1] = L2w_upper;
	L2w_threshold[0] = L2w_th_low;
	L2w_threshold[1] = L2w_th_upper;
}

double XPS_I2T::L2weight(void) { // mmps
	double w = L2w[0];
	return w;
	/*
	if (_I2T > _I2TRMS) {
		w += (L2w[1] - L2w[0]) * sqrt(_I2T / _I2TPeak);
		if (w > L2w[1]) w = L2w[1];
	}	
	return w;
	*/
}

double XPS_I2T::L2weight_Err(double Err_abs) { // mmps
	double w = L2w[1];
	if (Err_abs >= L2w_threshold[1]) {
		w = L2w[0];
	}
	else if (Err_abs <= L2w_threshold[0]) {
		w = L2w[1];
	}
	else {
		w = (L2w[1] - L2w[0]) / (L2w_threshold[1] - L2w_threshold[0]) * (Err_abs - L2w_threshold[0]) + L2w[0];
	}
	return w;
}


double XPS_I2T::update(double _acc) { // mmps2
	//updateI2TBuffer(_acc);
	updateI2T_ratio(_acc);
	return _I2T;
}
double XPS_I2T::updateI2T_ratio(double acc) {
	double I2T = acc*_den*acc;
	_I2T = (_I2T*(_I2T_BufferCountMax - 1)/_I2T_BufferCountMax + I2T);
	updatecurlimit();
	return _I2T;
}

double XPS_I2T::updateI2TBuffer(double acc) {
	_I2T_Buffer[_I2T_BufferCount] = acc*_den*acc;
	_sum += _I2T_Buffer[_I2T_BufferCount];
	_I2T_BufferCount = (_I2T_BufferCount++) % _I2T_BufferCountMax;
	_sum -= _I2T_Buffer[_I2T_BufferCount];
	_I2T = _sum;
	updatecurlimit();
	return _I2T;
}
void XPS_I2T::updatecurlimit(void) {
	counter_totaltime++;
	if (!currentlimitedmodeenabled) { // normal operation
		if (_I2T > _I2TRMS) {
			AccelerationLimit = AccelerationLimit_low;
			currentlimitedmodeenabled = true;
			if (counter_totaltime > 0)
				dutycycle = (double)((float)(counter_totaltime - counter_recoverytime) / (float)counter_totaltime);
			else
				dutycycle = 0;
			counter_totaltime = 0;
			counter_recoverytime = 0;
			//if (dutycycle > (1 - 1/120)-0.0001)
			//	set_AccelerationLimit_normal(AccelerationLimit_normal*1.01);
			//else 
			if (dutycycle < (1 - 1 / 60))
				set_AccelerationLimit_normal(AccelerationLimit_normal*0.98);
		}
		if (counter_totaltime_max <= counter_totaltime) {
			set_AccelerationLimit_normal(AccelerationLimit_normal*1.01);
			if (counter_totaltime > 0)
				dutycycle = (double)((float)(counter_totaltime - counter_recoverytime) / (float)counter_totaltime);
			else
				dutycycle = 0;
			counter_totaltime = 0;
			counter_recoverytime = 0;
			counter_totaltime_max = 250 * 60 * 2;
			set_AccelerationLimit_normal(AccelerationLimit_normal*1.01); 
		}
	}
	else { // during current limited mode
		counter_recoverytime++;
		if (_I2T < _I2TRMS*0.9) {
			AccelerationLimit = AccelerationLimit_normal;
			currentlimitedmodeenabled = false;
			counter_totaltime_max = counter_recoverytime * 120;
		}
	}
}
void XPS_I2T::set_AccelerationLimit_normal(double acc_normal) {
	if (acc_normal > _AccelerationLimit_max)
		AccelerationLimit_normal = _AccelerationLimit_max;
	else if (acc_normal < _AccelerationLimitRMS)
		AccelerationLimit_normal = _AccelerationLimitRMS;
	else
		AccelerationLimit_normal = acc_normal;
}

void XPS_I2T::set_limitCurrent(double _AccelerationLimit_low) {
	if (_AccelerationLimit_low > _AccelerationLimitRMS)
		AccelerationLimit_low = _AccelerationLimit_low;
	else
		AccelerationLimit_low = _AccelerationLimitRMS;
}
void XPS_I2T::clear(void) {
	for (int i = 0; i < _I2T_BufferCountMax; i++)
		_I2T_Buffer[i] = 0;
	_I2T = 0;
}
double XPS_I2T::I2T(void) {
	return _I2T;
}

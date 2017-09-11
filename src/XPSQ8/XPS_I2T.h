#ifndef STAGEI2T_H
#define STAGEI2T_H

#define I2TDriverMaximumCurrent 5.0
#define I2TDriverMaximumPeakCurrent 3.9
#define I2TDriverMaximumRMSCurrent 1.32
#define I2TDriverRMSIntegrationTime 10.0
#define I2TDriverMaximumPeakCurrentTime 4.0
#define ScalingAcc_X 5300.762
#define ScalingAcc_Y 13063.206

class XPS_I2T
{
public:
    XPS_I2T();
    ~XPS_I2T();
	double I2T(void);
	double I2TMax(void);
	void set(double AccelerationLimit, double dt, double L2w_low, double L2w_upper, double L2w_th_low, double L2w_th_upper);
	void set_L2w(double L2w_low, double L2w_upper, double L2w_th_low, double L2w_th_upper);
	void set_limitCurrent(double _AccelerationLimit_low);
	void set_AccelerationLimit_normal(double acc_normal);
	double update(double value);
	void clear(void);
	double L2weight(void);
	double L2weight_Err(double Err);
	double L2w[2];
	double L2w_threshold[2];
	double AccelerationLimit;
	double AccelerationLimit_normal;
	bool currentlimitedmodeenabled;
	double dutycycle;
private:
	double updateI2TBuffer(double newI2T);
	double updateI2T_ratio(double newI2T);
	void updatecurlimit(void);
	double _dt;
	double _AccelerationLimit_max;
	double AccelerationLimit_low;
	double _AccelerationLimitPeak;
	double _AccelerationLimitRMS;
	double _I2T;
	double _I2T_MAX;
	double * _I2T_Buffer;
	int _I2T_BufferCount;
	int _I2T_BufferCountMax;
	double _I2TPeak;
	double _I2TRMS;
	double _den;
	double _sum;
	uint64_t counter_recoverytime;
	uint64_t counter_totaltime;
	uint64_t counter_totaltime_max;
};

#endif // SERIALINTERFACEREADER_H

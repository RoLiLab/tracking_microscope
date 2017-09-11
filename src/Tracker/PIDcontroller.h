#ifndef _PIDcontroller_H
#define _PIDcontroller_H

#include "Base/base.h"


class PIDcontroller
{
public:
	PIDcontroller(void);
	~PIDcontroller(void);
	void updatePIDparameter(double _pk, double _pi, double _pd);
	double getDesiredPos(void);
	void setDesiredPos(double _Pos_desired);
	double getCurrentPos(void);
	void setCurrentPos(double _pos);
	double computeNewInputPID(void);
	double computeNewInputPID_Round(void);
	bool Pos_desired_set;
	bool Enable;
	void clear(void);
private:
	double Pos;
	double Pos_desired;
	double kp;
	double ki;
	double kd;
	double k1;
	double k2;
	double k3;
	vector<double> err;
};

#endif

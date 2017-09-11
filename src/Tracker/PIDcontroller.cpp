#include "Base/base.h"
#include "PIDcontroller.h"


// -----------------PID ----------------------------------------------------
PIDcontroller::PIDcontroller(void)  {
	Pos = 0;
	Pos_desired = 0;
	Pos_desired_set = false;
	kp = 20;
	ki = 0;
	kd = 0;
	k1 = 0;
	k2 = 0;
	k3 = 0;
	err.push_back(0);
	err.push_back(0);
	err.push_back(0);
	Enable = false;
};
PIDcontroller::~PIDcontroller(void)  {
};

void PIDcontroller::clear(void) {
	for (int i = 0; i < err.size(); i++)
		err[i] = 0;
	Pos_desired_set = false;
};
void PIDcontroller::updatePIDparameter(double _kp, double _ki, double _kd) {
	kp = _kp;
	ki = _ki;
	kd = _kd;
	if (_kd == 0) {
		k1 = kp;
		k2 = 0;
		k3 = 0;
	}
	else {
		k1 = kp + ki + kd;
		k2 = -kp - 2 * ki;
		k3 = kd;
	}
	if (kp == 0)
		Enable = false;

}
void PIDcontroller::setDesiredPos(double _Pos_desired) {
	Pos_desired_set = true;
	Pos_desired = _Pos_desired;
}

double PIDcontroller::getDesiredPos(void) {
	return Pos_desired;
}
void PIDcontroller::setCurrentPos(double _pos) {
	Pos = _pos;
}

double PIDcontroller::getCurrentPos(void) {
	return Pos;
}

double PIDcontroller::computeNewInputPID(void) {
	if (Pos_desired_set) {
		err.insert(err.begin(), Pos_desired - Pos); err.pop_back();
		return k1*err[0] + k2*err[1] + k3*err[2];
	}
	return 0;
}

double PIDcontroller::computeNewInputPID_Round(void) {
	double pxDist = 0.015 * 2;
	if (Pos_desired_set) {
		double err_cur = Pos_desired - Pos;
		if (fabs(err_cur) <= pxDist) err_cur = 0;
		err.insert(err.begin(), err_cur); err.pop_back();
		return k1*err[0] + k2*err[1] + k3*err[2];
	}
	return 0;
}

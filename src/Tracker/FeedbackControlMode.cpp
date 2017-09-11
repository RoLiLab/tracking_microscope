#include "Base/base.h"
#include "Tracker/FeedbackControlMode.h"


// --------------------------------------------------------------------------------
FeedbackControlMode::FeedbackControlMode(void)
{
	double _maximumfollowingerror = 1.0; // when the error between MPC open loop position and the current stage position is greater than 1mm, b_followingerrorfailure becomes true
	int _followingerrorcooldownframeset = 2500; // 2500 frame = 10 seconds
	int _maximumfollowingerror_countMax = 5;
	updateParameters(_maximumfollowingerror, _followingerrorcooldownframeset, _maximumfollowingerror_countMax);
	reset();
};
FeedbackControlMode::~FeedbackControlMode(void)  {
};

void FeedbackControlMode::reset(void) {
	b_istracking = false;
	b_isMPCcontrol = false; // true: MPC mode (open loop),  false: PID mode
	b_followingerrorfailure = false;
	followingerrorcooldownframe = 0;
	maximumfollowingerror_count = 0;
}

void FeedbackControlMode::updateParameters(double _maximumfollowingerror, int _followingerrorcooldownframeset, int _maximumfollowingerror_countMax) {
	maximumfollowingerror = _maximumfollowingerror; // when the error between MPC open loop position and the current stage position is greater than 1mm, b_followingerrorfailure becomes true
	followingerrorcooldownframeset = _followingerrorcooldownframeset; // 2500 frame = 10 seconds
	maximumfollowingerror_countMax = _maximumfollowingerror_countMax;
}
void FeedbackControlMode::checkingfollowingerror_I2T(double i2t) {
	if (!b_followingerrorfailure) { // normal mode
		if (i2t > 10.0) {
			maximumfollowingerror_count++;
			if (maximumfollowingerror_count > maximumfollowingerror_countMax) {
				b_followingerrorfailure = true;
				followingerrorcooldownframe = followingerrorcooldownframeset;
				maximumfollowingerror_count = 0;
			}
		}
		else {
			if (maximumfollowingerror_count > 0)
				maximumfollowingerror_count--;
		}
	}
	else { // failure mode
		followingerrorcooldownframe--;
		if (followingerrorcooldownframe <= 0) {
			b_followingerrorfailure = false;
		}
	}
}
void FeedbackControlMode::checkingfollowingerror(double * ref, double * cur) {
	double dx = fabs(cur[0] - ref[0]);
	double dy = fabs(cur[1] - ref[1]);
	if (!b_followingerrorfailure) { // normal mode
		if (dx > maximumfollowingerror || dy > maximumfollowingerror) {
			maximumfollowingerror_count++;
			if (maximumfollowingerror_count > maximumfollowingerror_countMax) {
				b_followingerrorfailure = true;
				followingerrorcooldownframe = followingerrorcooldownframeset;
				maximumfollowingerror_count = 0;
			}
		}
		else {
			if (maximumfollowingerror_count > 0)
				maximumfollowingerror_count--;
		}
	}
	else { // failure mode
		followingerrorcooldownframe--;
		if (followingerrorcooldownframe <= 0) {
			b_followingerrorfailure = false;
		}
	}
}

void FeedbackControlMode::enabletracking(bool _signal) {
	b_istracking = _signal;
}
void FeedbackControlMode::enableMPC(bool _signal){
	b_isMPCcontrol = _signal;
}
bool FeedbackControlMode::istracking(void){
	return b_istracking;
}
bool FeedbackControlMode::isMPC(void){
	//return (b_isMPCcontrol && (!b_followingerrorfailure));
	return b_isMPCcontrol;
}
bool FeedbackControlMode::isfollowingerrorfailed(void){ // b_followingerrorfailure
	return b_followingerrorfailure;
}

double FeedbackControlMode::getmaximumfollowingerror(void){
	return maximumfollowingerror;
}
void FeedbackControlMode::setmaximumfollowingerror(double _value){
	maximumfollowingerror = _value;
}
double FeedbackControlMode::getfollowingerrorcooldownframeset(void){
	return followingerrorcooldownframeset;
}
void FeedbackControlMode::setfollowingerrorcooldownframeset(int _value){
	followingerrorcooldownframeset = _value;
}

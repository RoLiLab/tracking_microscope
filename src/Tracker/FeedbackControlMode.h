#ifndef _FeedbackControlMode_H
#define _FeedbackControlMode_H

#include "Base/base.h"


class FeedbackControlMode
{
public:
	FeedbackControlMode(void);
	~FeedbackControlMode(void);
	bool b_istracking;
	bool b_isMPCcontrol; // true: MPC mode (open loop),  false: PID mode
	bool b_followingerrorfailure;
	int followingerrorcooldownframe;

	void checkingfollowingerror(double * ref, double * cur);
	void checkingfollowingerror_I2T(double i2t);

	void enabletracking(bool _signal);
	void enableMPC(bool _signal);
	bool istracking(void);
	bool isMPC(void);
	bool isfollowingerrorfailed(void); // b_followingerrorfailure
	double getmaximumfollowingerror(void);
	void setmaximumfollowingerror(double _value);
	double getfollowingerrorcooldownframeset(void);
	void setfollowingerrorcooldownframeset(int _value);
	void reset(void);
	void updateParameters(double _maximumfollowingerror, int _followingerrorcooldownframeset, int _maximumfollowingerror_countMax);
private:
	double maximumfollowingerror;
	int maximumfollowingerror_count;
	int maximumfollowingerror_countMax;
	int followingerrorcooldownframeset;
};

#endif

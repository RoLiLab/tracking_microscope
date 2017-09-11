#ifndef _ZEBRAFISHFISHPREDICTOR_H
#define _ZEBRAFISHFISHPREDICTOR_H

#include "Base/base.h"
#include "Tracker/TrackingInfo.h"
#include "mkl_lapacke.h"
#include "mkl_cblas.h"

class buffer_double
{
	// new data at the beginning, the last data is the oldest one;
public:
	buffer_double(void);
	~buffer_double(void);
	void resetvalue(double v);
	void resetsize(int n);
	void reset(int n, double v);

	void update(double v);
	double mean(void);
	void update_rad(double v); // generate continuous data in radian
	void update_deg(double v); // generate continuous data in radian

	int step;
	double * data;

};

double adjustAngle(double v_prev, double v_next, double range);
void adjustAngle(double * v, int n, double range);

class zebrafishpredictor
{
public:
	zebrafishpredictor(void);
	~zebrafishpredictor(void);
	buffer_double x;
	buffer_double y;
	buffer_double fshvel;
	buffer_double th;
	buffer_double xproj;
	buffer_double yproj;
	buffer_double thfiltered;
	buffer_double thfiltered_COS;
	buffer_double thfiltered_SIN;
	buffer_double weight;
	buffer_double vparallel;
	buffer_double vppd;
	buffer_double weightedVel;
	double vperpendicular;
	double vfish;
	double thetamoving;
	double dthetabtMovingHeading;
	double vslopepred;
	bool isFishMoving;

	double * predX; // [0] = current , [buffersize - 1] -> far future
	double * predY;
	int steps;

	void reset(int _halfcycle_ms, int _dt_ms, int _predStep);
	void resetParameter(double _weightThresholdMin, double _weightThresholdMax, double _terminalVelocity, double _slopegain);
	void init(double _x, double _y, double _th);
	void update(double _x, double _y, double _th);

	void updatePos(double _x, double _y, double _th);

private:
	int halfcycle_ms;
	int dt_ms;
	double dt_sec;
	int bufferSize;
	bool updated;

	double weightThresholdMin;
	double weightThresholdMax;
	double terminalVelocity;
	double gainvslopepred;

	void getThetacenterMinMax(void);
	void getThetaSINCOS(void);
	void getProjPt(void);
	void getWeight(void);
	double computeWeight_Linear(double phi);
	void getWeightedVel(void);
	double getVelSlopePred(void);
	void predictFishFuturePosition(void);
};

double standard_deviation(double * data, int n);

#endif

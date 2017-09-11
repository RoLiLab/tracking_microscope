#ifndef _FISHPREDICTOR_H
#define _FISHPREDICTOR_H

#include "KBase/common/common.h"
#include "Tracker/TrackingInfo.h"

class fishposPredictor
{
public:
	fishposPredictor(void);
	~fishposPredictor(void);

	void clear(void);


private:
	double dt_sec;
	int pred_steps;
	int posbuf_steps;
	int velbuf_steps;
	double * x_buf;
	double * y_buf;
	double * vx_buf;
	double * vy_buf;
	double th_LPF;
	double alpha_LPF;
	double freq_LPF;
	// output
	double * x_pred;// from step
	double * y_pred; // from step
	double * vx_pred; // from step
	double * vy_pred; // from step
	double x_proj; // projected point x from the given x and y
	double y_proj; // projected point y from the given x and y
	double u_p[2]; // unit vector parallel
	double u_v[2]; // unit vector vertical

	void ProjPt(void); // --> update x_proj, y_proj
	void UnitVec(double th); // --> update u_p, u_v
	void computeth_LPF(double th); // --> update th_LPF
	void

};

#endif

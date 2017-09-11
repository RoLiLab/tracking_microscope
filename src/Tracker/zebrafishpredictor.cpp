#include "Base/base.h"
#include "Tracker/zebrafishpredictor.h"

// --------------------------------------------------------------------------------
buffer_double::buffer_double(void)
{
	step = 0;
	data = NULL;
	reset(6, 0);
};
buffer_double::~buffer_double(void)  {
	if (data)
		free(data);
};
void buffer_double::resetsize(int n)  {
	if (data)
		free(data);
	step = n;
	data = (double *) malloc (step * sizeof(double));
};
void buffer_double::resetvalue(double v)  {
	for (int i = 0; i < step; i++)
		data[i] = v;
};
void buffer_double::reset(int n, double v) {
	if (n != step)
		resetsize(n);
	resetvalue(v);
}
void buffer_double::update(double v)  {
	for (int i = step; i > 0; i--)
		data[i] = data[i-1];
	data[0] = v;
};


double buffer_double::mean(void)  {
	double mean = 0;
	for (int i = step; i > 0; i--)
		mean = fabs(data[i]);
	mean /= step;
	return mean;
};
void buffer_double::update_rad(double v) {
	double v_adjusted = adjustAngle(data[0], v, PI);
	update(v_adjusted);
}
void buffer_double::update_deg(double v) {
	double v_adjusted = adjustAngle(data[0], v, 180);
	update(v_adjusted);
}

double adjustAngle(double v_prev, double v_next, double range) {
	double dv = 2*range;
	if (v_next > v_prev) {
		dv = -dv;
	}
	while (fabs(v_next - v_prev ) > range)
		v_next += dv;
	return v_next;
}
void adjustAngle(double * v, int n, double range) {
	for (int i = 0; i < n-1; i++)
		v[i+1] = adjustAngle(v[i], v[i+1], range);
}
// --------------------------------------------------------------------------------
zebrafishpredictor::zebrafishpredictor(void)
{
	predX = NULL;
	predY = NULL;
	resetParameter(PI/15, PI/6, 1.0, 5.0);
	reset(24, 4, 10);
};
zebrafishpredictor::~zebrafishpredictor(void)  {
	if (predX) free(predX);
	if (predY) free(predY);
};

void zebrafishpredictor::reset(int _halfcycle_ms, int _dt_ms, int _predStep) {
	halfcycle_ms = _halfcycle_ms;
	dt_ms = _dt_ms;
	dt_sec = (double)dt_ms / 1000;
	steps = _predStep;
	bufferSize = halfcycle_ms/_dt_ms;
	if (bufferSize < 3) bufferSize = 3;
	fshvel.reset(bufferSize, 0.0);
	x.reset(bufferSize, 0.0);
	y.reset(bufferSize, 0.0);
	th.reset(bufferSize, 0.0);
	xproj.reset(bufferSize, 0.0);
	yproj.reset(bufferSize, 0.0);
	thfiltered.reset(bufferSize, 0.0);
	weight.reset(bufferSize, 1.0);
	vparallel.reset(bufferSize, 0.0);
	weightedVel.reset(bufferSize, 0.0);
	thfiltered_COS.reset(bufferSize, 1.0);
	thfiltered_SIN.reset(bufferSize, 0.0);
	vperpendicular = 0;
	vslopepred = 0;
	isFishMoving = false;
	if (predX) free(predX);
	if (predY) free(predY);
	predX = (double *) malloc (steps * sizeof(double));
	predY = (double *) malloc (steps * sizeof(double));
	updated = false;
}

void zebrafishpredictor::resetParameter(double _weightThresholdMin, double _weightThresholdMax, double _terminalVelocity, double _slopegain) {
	weightThresholdMin = _weightThresholdMin;
	weightThresholdMax = _weightThresholdMax;
	terminalVelocity = _terminalVelocity; //mmps
	gainvslopepred = _slopegain;
}
void zebrafishpredictor::init(double _x, double _y, double _th) {
	x.resetvalue(_x);
	y.resetvalue(_y);
	th.resetvalue(_th);
	xproj.resetvalue(_x);
	yproj.resetvalue(_y);
	thfiltered.resetvalue(_th);
	getThetaSINCOS();
	updated = true;
}

void zebrafishpredictor::update(double _x, double _y, double _th) {
	if (updated) {
		// 1. update pos
		updatePos(_x, _y, _th); // update x, y, th
		// 2. filtered position (th (& heading vector) -> x, y)
		getThetacenterMinMax(); // update thfiltered, thfiltered_COS, thfiltered_SIN
		getProjPt(); // update xproj, yproj
		// 3. compute weight
	}
	else {
		init(_x, _y, _th);
	}
	getWeight(); // update weight, vparallel, vperpendicular, vfish, thetamoving, dthetabtMovingHeading
	//getWeightedVel(); // update weightedVel
	weightedVel.update(vparallel.mean());

	vslopepred = getVelSlopePred(); // update vslopepred
	predictFishFuturePosition(); // update xPred, yPred

	if (vparallel.data[0] > 10)
		isFishMoving = true;
}

void zebrafishpredictor::updatePos(double _x, double _y, double _th) {
	x.update(_x);
	y.update(_y);
	th.update_deg(_th);
	double dx = x.data[0] - x.data[1];
	double dy = y.data[0] - y.data[1];
	fshvel.update(sqrt(dx*dx + dy*dy)*250);
}

void zebrafishpredictor::getThetacenterMinMax(void) {
	double _min = th.data[0];
	double _max = th.data[0];
	for (int i = 1; i < th.step; i++) {
		if (th.data[i] > _max)
			_max = th.data[i];
		if (th.data[i] < _min)
			_min = th.data[i];
	}
	double thCenterMinMax = (_min + _max)/2;
	thfiltered.update_deg(thCenterMinMax);
	getThetaSINCOS();
}

void zebrafishpredictor::getThetaSINCOS(void) {
	double thRad = thfiltered.data[0]*PI/180;
	thfiltered_COS.update(cos(thRad));
	thfiltered_SIN.update(sin(thRad));
}

void zebrafishpredictor::getProjPt(void) {
	double cosH = thfiltered_COS.data[0];
	double sinH = thfiltered_SIN.data[0];
	double * xo = (double *) malloc (bufferSize * sizeof(double));
	double yoSum = 0;
	for (int i = 0; i < bufferSize; i++) {
		xo[i] = cosH * x.data[i] + sinH * y.data[i];
		yoSum += (-sinH * x.data[i] + cosH * y.data[i]);
	}
	double yoMean = yoSum / bufferSize;
	xproj.update(cosH * xo[0] - sinH * yoMean);
	yproj.update(sinH * xo[0] + cosH * yoMean);
	free(xo);
};

void zebrafishpredictor::getWeight(void) {
	//compute a swimming direction (not heading direction)
	double cosH = thfiltered_COS.data[0];
	double sinH = thfiltered_SIN.data[0];
	double dx = xproj.data[1] - xproj.data[0];
	double dy = yproj.data[1] - yproj.data[0];
	vparallel.update((dx*cosH + dy*sinH)/dt_sec);
	vperpendicular = (-dx*sinH + dy*cosH)/dt_sec;
	vppd.update(vperpendicular);
	vfish = sqrt(dx*dx + dy*dy)/dt_sec;
	if (dx == 0)
		thetamoving = thfiltered.data[0];
	else
		thetamoving = adjustAngle(thfiltered.data[0], atan2(dy, dx)*180/PI, 180);
	//compute an angle difference b/t heading and swimming direction)
	if ((vparallel.data[0] < terminalVelocity) && (sqrt(dx*dx + dy*dy) < terminalVelocity))
		dthetabtMovingHeading = 0;
	else
		dthetabtMovingHeading = fabs(thetamoving - thfiltered.data[0]);

	// compute weight w
	if ((vparallel.mean() > vppd.mean()) && (vparallel.mean()>0.5)) // ( ||
		weight.update(computeWeight_Linear(dthetabtMovingHeading));
		//weight.update(1.0);
	else
		weight.update(0);
}
double zebrafishpredictor::computeWeight_Linear(double phi_deg) {
	double phi = phi_deg * PI / 180;
    double _w;
	if (phi > weightThresholdMax)
        _w = 0;
    else if (phi < weightThresholdMin)
        _w = 1;
    else
        _w = 1/(weightThresholdMin - weightThresholdMax) * (phi - weightThresholdMax);
	return _w;
}

void zebrafishpredictor::getWeightedVel(void) {
	double w_sum = 0;
	double vw_sum = 0;
	for (int i = 0; i < bufferSize; i++) {
		w_sum += weight.data[i];
		vw_sum += weight.data[i]*vparallel.data[i];
	}
	double vw;
	if (w_sum == 0)
		vw = 0;
	else
		vw = vw_sum/w_sum;
	if (vw < 0)
		vw = 0;
	weightedVel.update(vw);
}

double zebrafishpredictor::getVelSlopePred(void) {
	MKL_INT m, n, k, nrhs, lda, ldb, ldc, info; double alpha, beta;

	double * wV = (double *) malloc (bufferSize * sizeof(double));
	double * A = (double *) malloc (2*bufferSize * sizeof(double));
	memcpy(wV, weightedVel.data, bufferSize*sizeof(double));
	for (int i = 0; i < bufferSize; i++) {
		A[2*i] = 1;
		A[2*i+1] = bufferSize - i;
	}
	m = bufferSize; n = 2; nrhs = 1; lda = n; ldb = nrhs;

	LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, A, lda, wV, ldb );
	double slope = wV[1];
	if (slope > 0) slope = 0;

	for (int i = 0; i < bufferSize; i++)
		A[i] = (weightedVel.data[i]- (bufferSize - i)*wV[1] + wV[0]);

	double _std = standard_deviation(A, bufferSize);
	if (_std > 0.4)
		slope = 0;
	else
		int temp = 0;

	free(A); free(wV);
	return slope;
}

double standard_deviation(double * data, int n)
{
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/n);
}


void zebrafishpredictor::predictFishFuturePosition(void) {
	double cosH = thfiltered_COS.data[0];
	double sinH = thfiltered_SIN.data[0];
	if (weightedVel.data[0] > 0)
		for (int i = 0; i < steps; i++) {
			//double curVel = weightedVel.data[0] + gainvslopepred*(i)*vslopepred;
			double curVel = weightedVel.data[0] * gainvslopepred;
			if (i == 0) {
				predX[i] = xproj.data[0] - curVel*cosH*dt_sec;
				predY[i] = yproj.data[0] - curVel*sinH*dt_sec;
				//predX[i] = x.data[0] - curVel*cosH*dt_sec;
				//predY[i] = y.data[0] - curVel*sinH*dt_sec;
			}
			else {
				predX[i] = predX[i-1] - curVel*cosH*dt_sec;
				predY[i] = predY[i-1] - curVel*sinH*dt_sec;
			}
		}
	else
	{
		for (int i = 0; i < steps; i++) {
			predX[i] = x.data[0];
			predY[i] = y.data[0];
		}
	}

}

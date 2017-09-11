#include "KBase/common/common.h"
#include "Tracker/fishposPredictor.h"


// --------------------------------------------------------------------------------
fishposPredictor::fishposPredictor(void)
{
	dt_sec = 0.004;
	pred_steps = 5;
	posbuf_steps = 5;
	velbuf_steps =  5;
	x_buf = NULL;
	y_buf = NULL;
	vx_buf = NULL;
	vy_buf = NULL;
	th_LPF = 0;
	alpha_LPF = 0;
	freq_LPF = 0.2;
	// output
	x_pred = NULL;// from step
	y_pred = NULL; // from step
	vx_pred = NULL; // from step
	vy_pred = NULL; // from step
	x_proj = 0; // projected point x from the given x and y
	y_proj = 0; // projected point y from the given x and y
	u_p[0] = 0; u_v[0]; u_p[1] = 0; u_v[1]

};
fishposPredictor::~fishposPredictor(void)  {
	clear();
};


void fishposPredictor::clear(void) {
	if (x_buf) free(x_buf);
	if (y_buf) free(y_buf);
	if (vx_buf) free(vx_buf);
	if (vy_buf) free(vy_buf);
	if (x_pred) free(x_pred);
	if (y_pred) free(y_pred);
	if (vx_pred) free(vx_pred);
	if (vy_pred) free(vy_pred);
}

void fishposPredictor::reset(void) {

}


void MPC_XPSAVT::setVelMax(double _v) {velMax = _v;}
double MPC_XPSAVT::getVelMax(void) {return velMax;}
void MPC_XPSAVT::setWeightL1(double  _w) {weight_L1 = _w;}
double MPC_XPSAVT::getWeightL1(void) {return weight_L1;}
void MPC_XPSAVT::setWeightL2(double  _w) {weight_L2 = _w;}
double MPC_XPSAVT::getWeightL2(void) {return weight_L2;}
void MPC_XPSAVT::setCntrHrz(int _steps) {cntrHrz = _steps;}
int MPC_XPSAVT::getCntrHrz(void) {return cntrHrz;}
void MPC_XPSAVT::setPredHrz(int _steps) {predHrz = _steps;}
int MPC_XPSAVT::getPredHrz(void) {return predHrz;}
void MPC_XPSAVT::terminate(void) {isInit = false;}
bool MPC_XPSAVT::isInitiated(void) {return isInit;}
double MPC_XPSAVT::getCurVel(void) {if (u != NULL)	return u[0]; else return 0;}
double MPC_XPSAVT::getCurPos(void) {if (p_precommitted != NULL)	return p_precommitted[0]; else return 0;}
double * MPC_XPSAVT::getP_precommitted(void) {return p_precommitted;}
double * MPC_XPSAVT::getVelHistory(void) {return u;}
double * MPC_XPSAVT::getStpResp(void) {return stpRsp;}
double * MPC_XPSAVT::getImpResp(void) {return impRsp;}
void MPC_XPSAVT::updateParameters(int _cntrHrz, double _wL1, double _wL2, double _velMax) {
	setCntrHrz(_cntrHrz);	setWeightL1(_wL1);	setWeightL2(_wL2);	setVelMax(_velMax);
}
void MPC_XPSAVT::updateImpRsp(int n, double * _impRsp) {
	if (n <= 0) return;
	if (impRsp) free(impRsp);
	if (stpRsp) free(stpRsp);
	predHrz = n;
	impRsp = (double *) malloc (predHrz * sizeof(double));
	stpRsp = (double *) malloc (predHrz * sizeof(double));

	impRsp[0] = _impRsp[0];
	stpRsp[0] = 0;
	for (int i = 1; i < predHrz ; i++) {
		impRsp[i] = _impRsp[i];
		stpRsp[i] = stpRsp[i-1] + impRsp[i];
	}
}
void MPC_XPSAVT::generateMat(void) {
	// Prepare A (weighted)
	if (A) free(A);
	int rows_A = predHrz;
	int rows_AL1 = cntrHrz;
	int rows_AL2 = cntrHrz-1;
	int cols = cntrHrz;
	A = (double *) malloc (cols * (rows_A + rows_AL1 + rows_AL2)* sizeof(double));
	// A(predHrz); L1 (contrHrz); L2(contrHrz - 1)
	for (int j = 0; j < cols; j++) {
		for (int i = 0; i < rows_A; i++) {
			int ind = i - j;
			if (ind < 0)
				A[i*cols+j] = 0;
			else
				A[i*cols+j] = stpRsp[ind];
		}
		for (int i = rows_A; i < rows_A + rows_AL1 +rows_AL2; i++)
			A[i*cols+j] = 0;
	}
	if (B) free(B);
	B = (double *) malloc ((rows_A + rows_AL1 + rows_AL2)* sizeof(double));

	if (p_precommitted) free(p_precommitted);
	p_precommitted = (double *) malloc (predHrz* sizeof(double));

	if (u) free(u);
	u = (double *) malloc (predHrz* sizeof(double));
}
void MPC_XPSAVT::setL1Mat(double _wL1) {
	setWeightL1(_wL1);
	if (A == NULL) return;
	double * L1 = &A[cntrHrz * predHrz];
	int cols = cntrHrz;
	int rows = cntrHrz;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (i == j)
				L1[i*cols + j] = weight_L1;
			else
				L1[i*cols + j] = 0;
		}
	}
}
void MPC_XPSAVT::setL2Mat(double _wL2) {
	setWeightL2(_wL2);
	if (A == NULL) return;
	double * L1 = &A[cntrHrz * (predHrz + cntrHrz)];
	int cols = cntrHrz;
	int rows = cntrHrz - 1;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (i == j)
				L1[i*cols + j] = weight_L2;
			else if (i + 1== j)
				L1[i*cols + j] = -weight_L2;
			else
				L1[i*cols + j] = 0;
		}
	}
}
void MPC_XPSAVT::copyDesiredPos(int n, double * _desiredPos) {
	int rows_A = predHrz;
	int rows_AL = predHrz + cntrHrz + cntrHrz-1;

	for (int i = 0; i < rows_A - 1; i++)
		p_precommitted[i] = p_precommitted[i + 1];

	int i = 0;
	for (i = 0; i < n; i++)
		B[i] = _desiredPos[i] - p_precommitted[i];
	for (; i < rows_A; i++)
		B[i] = _desiredPos[n-1] - p_precommitted[i];
	for (; i< rows_AL; i++)
		B[i] = 0;
}
void MPC_XPSAVT::update_p_precommoitted(double vel) {
	for (int i = 0; i < predHrz; i++)
		p_precommitted[i] += stpRsp[i]*vel;
	for (int i = 0; i < predHrz - 1; i++)
		u[i] = u[i + 1];
	u[predHrz - 1] = vel; // just for update history. not related for the calculation
}
void MPC_XPSAVT::step_weight(int n, double * p, double _wL1, double _wL2, double * nextPos, double * nextVel) {
	setL1Mat(_wL1);
	setL2Mat(_wL2);
	copyDesiredPos(n, p);
	MPC();
	getVel_limitted(B);
	update_p_precommoitted(B[0]);
	*nextPos = p_precommitted[0];
	*nextVel = B[0];
}
void MPC_XPSAVT::step(int n, double * p, double * nextPos, double * nextVel) {
	setL1Mat(weight_L1);
	setL2Mat(weight_L2);
	copyDesiredPos(n, p);
	MPC();
	getVel_limitted(B);
	update_p_precommoitted(B[0]);
	*nextPos = p_precommitted[0];
	*nextVel = B[0];
}


void MPC_XPSAVT::MPC(void) {
	MKL_INT m, n, k, nrhs, lda, ldb, ldc, info; double alpha, beta;
	int rows_A = predHrz;
	int rows_AL1 = cntrHrz;
	int rows_AL2 = cntrHrz-1;
	int cols = cntrHrz;
	m = predHrz + cntrHrz + cntrHrz -1; n = cntrHrz; nrhs = 1; lda = n; ldb = nrhs;

	//LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, A, lda, B, ldb );

	double * tempBuf = (double *) malloc (m * n * sizeof(double));
	memcpy(tempBuf, A, (m * n)*sizeof(double));
	LAPACKE_dgels( LAPACK_ROW_MAJOR, 'N', m, n, nrhs, tempBuf, lda, B, ldb );
	free(tempBuf);

	return;
}


void MPC_XPSAVT::getVel_limitted(double * _vel) {
	if (*_vel > velMax) *_vel = velMax;
	else if (*_vel < -velMax) *_vel = -velMax;
}
void MPC_XPSAVT::replaceNextInput(double _v) {
	for (int i = 0; i < predHrz; i++)
		p_precommitted[i] = p_precommitted[i] + stpRsp[i]*(_v - u[predHrz - 1]) ;
	u[predHrz - 1] = _v; // just for update history. not related for the calculation
}

bool MPC_XPSAVT::InitPos(double _initPos) {
	if (A == NULL) return false;
	for (int i = 0; i < predHrz; i++) {
		p_precommitted[i] = _initPos ;
		u[i] = 0;
	}
	isInit = true;
	return isInit;
}

void MPC_XPSAVT::Init(int _cntrHrz, double _wL1, double _wL2, double _velMax, int n, double * _impRsp) {
	updateParameters(_cntrHrz, _wL1, _wL2, _velMax);
	updateImpRsp(n, _impRsp);
	generateMat();
	isInit = false;
}

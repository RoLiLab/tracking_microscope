#include "Base/base.h"
#include "MPC.h"

// --------------------------------------------------------------------------------
MPC_XPSAVT::MPC_XPSAVT(void)
{
	impRsp_ALL = NULL;
	impRsp_ALL_n = 17;
	impRsp_ALL_stride = 40;

	impRsp = NULL;
	stpRsp = NULL;
	impRsp_Acc = NULL;
	u = NULL;
	p_precommitted = NULL;
	A = NULL;
	B = NULL;
	n_impRsp_Acc = 0;
	Acc_limit = 5000;
	Acc_limit_set = 5000;
	for (int i = 0; i < 4; i++)
		vel_limit_cur[i] = 0;
	int _predHrz = 101;
	int _cntrHrz = 10;
	double _weight_L1 = 0;
	double _weight_L2 = 0;
	double _velMax = 150;
	double _dvelMax = 80;
	double _impRsp[] = {0.000000000000000, 0.000000000000000, 0.000151724468335, 0.001373062470747, 0.001690015502758, 0.000753247051303, 0.000137722815986, -0.000142037092876, -0.000039199098455, -0.000034213548236, 0.000104658980903, 0.000011183502862, 0.000001239874051, -0.000019128721123, 0.000008841056127, 0.000017399588584, 0.000011624132795, 0.000001712507977, 0.000005477146452, -0.000009730167878, 0.000021070619405, -0.000027868137885, -0.000011574854740, 0.000001734908889, 0.000023810878032, 0.000002692799648, -0.000005084675688, -0.000017245322915, -0.000000775964176, 0.000009071169742, 0.000026287637141, -0.000009649151444, -0.000004832481351, -0.000011804483250, 0.000004412912983, 0.000003582418201, 0.000013215244379, 0.000003859317204, 0.000010022873328, -0.000037349233981, -0.000031626248462, -0.000001100335603, 0.000022158707541, 0.000012771106426, 0.000006480898469, -0.000004195000803, 0.000004260413101, 0.000005091772954, 0.000005369361525, -0.000000299843961, 0.000003487009911, -0.000010048564329, -0.000004076508580, -0.000008066444699, 0.000010347820735, 0.000008042059311, 0.000012849563972, -0.000017046813160, -0.000013355629240, 0.000001358072281, 0.000015646243858, 0.000009449788012, -0.000013118840684, -0.000016056769613, -0.000006346199409, -0.000011489249299, 0.000017320070788, 0.000025713709589, 0.000013047974810, -0.000007522822712, -0.000028306586873, -0.000006112791933, 0.000003411042580, 0.000002520461964, 0.000020870159088, 0.000010958453312, -0.000014307261613, -0.000023135190223, -0.000005508168394, 0.000011226199111, 0.000014290584550, 0.000005997577864, 0.000005388017090, -0.000006896581144, -0.000020894934116, -0.000008320528558, 0.000015737167188, 0.000016334483449, 0.000006128815570, 0.000003885022831, -0.000018605608183, -0.000023462079927, 0.000001925745425, 0.000013392339886, 0.000006130951585, 0.000005707789728, -0.000001012048616, -0.000000641354041, 0.000000448736003, 0.000008246040129, 0.000004597644093};
	Init(_cntrHrz, _weight_L1, _weight_L2, _velMax, _dvelMax, _predHrz, _impRsp);

};
MPC_XPSAVT::~MPC_XPSAVT(void)  {
	clear();
};
void MPC_XPSAVT::clear(void) {
	if (impRsp) free(impRsp);
	if (stpRsp) free(stpRsp);
	if (A) free(A);
	if (u) free(u);
	if (p_precommitted) free(p_precommitted);
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
void MPC_XPSAVT::updateParameters(int _cntrHrz, double _wL1, double _wL2, double _velMax, double _dvelMax) {
	setCntrHrz(_cntrHrz);	setWeightL1(_wL1);	setWeightL2(_wL2);	setVelMax(_velMax);
	dvelMax = _dvelMax;
}


void MPC_XPSAVT::updateImpRsp_Acc(int n, double * _impRsp_Acc, double _Acc_max) {
	if (n <= 0) return;
	if (impRsp_Acc) free(impRsp_Acc);
	Acc_limit = _Acc_max;
	Acc_limit_set = _Acc_max;
	n_impRsp_Acc = n;
	impRsp_Acc = (double *)malloc(n_impRsp_Acc * sizeof(double));
	for (int i = 0; i < n_impRsp_Acc; i++) {
		impRsp_Acc[i] = _impRsp_Acc[i];
	}
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

void MPC_XPSAVT::update_p_precommoitted2(double vel) {
	double * resp = NULL;
	if (impRsp_ALL == NULL) {
		for (int i = 0; i < predHrz; i++)
			p_precommitted[i] += stpRsp[i] * vel;
	}
	else {
		// impRsp_ALL_n = 17;
		// impRsp_ALL_stride = 40;
		// predHrz == 100
		int n1 = (floor(abs(vel / 10)));
		if (n1 == 0) {
			for (int i = 0; i < impRsp_ALL_stride; i++)
				p_precommitted[i] += impRsp_ALL[0 + i*impRsp_ALL_n] * vel;
		}
		else if (n1 >= impRsp_ALL_n) {
			for (int i = 0; i < impRsp_ALL_stride; i++)
				p_precommitted[i] += impRsp_ALL[(impRsp_ALL_n-1) + i*impRsp_ALL_n] * vel;
		}
		else {
			double alpha = (abs(vel) - (double)(n1)*10.0)/10;
			for (int i = 0; i < impRsp_ALL_stride; i++) {
				double x0 = impRsp_ALL[n1 - 1 + i*impRsp_ALL_n];
				double x1 = impRsp_ALL[n1 + i*impRsp_ALL_n];
				double x_impRsp = (1 - alpha)*x0 + alpha*x1;
				p_precommitted[i] += x_impRsp*vel;
			}
		}
		for (int i = impRsp_ALL_stride; i < predHrz; i++) {
			p_precommitted[i] = p_precommitted[impRsp_ALL_stride-1];
		}
	}
	for (int i = 0; i < predHrz - 1; i++)
		u[i] = u[i + 1];
	u[predHrz - 1] = vel; // just for update history. not related for the calculation
}

void MPC_XPSAVT::step_weight(int n, double * p, double _wL1, double _wL2, double * nextPos, double * nextVel) {
	setL1Mat(_wL1);
	setL2Mat(_wL2);
	copyDesiredPos(n, p);
	MPC();
	getVel_limitted_Acc(B);
	update_p_precommoitted2(B[0]);
	*nextPos = p_precommitted[0];
	*nextVel = B[0];
}
void MPC_XPSAVT::step(int n, double * p, double * nextPos, double * nextVel) {
	setL1Mat(weight_L1);
	setL2Mat(weight_L2);
	copyDesiredPos(n, p);
	MPC();
	getVel_limitted(B);
	update_p_precommoitted2(B[0]);
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
double MPC_XPSAVT::step_PID(double curpos, double desiredpos) {

	double pidvel = 0;
	PIDCntr.setCurrentPos(curpos);
	PIDCntr.setDesiredPos(desiredpos);
	pidvel = PIDCntr.computeNewInputPID();
	limit_PIDinput(&pidvel);
	return pidvel;
}

void MPC_XPSAVT::set_pidvel_limit(double _vmaxpid_upper, double _vmaxpid_lower) {
	vel_limit_cur[2] = _vmaxpid_upper;
	vel_limit_cur[3] = _vmaxpid_lower;
}

void MPC_XPSAVT::set_pidvel_limit_max(void) {
	vel_limit_cur[2] = velMax;
	vel_limit_cur[3] = -velMax;
}

void MPC_XPSAVT::limit_PIDinput(double * _pidvel) {

	// limit range from Acc
	if (*_pidvel > vel_limit_cur[2]) *_pidvel = vel_limit_cur[2];
	else if (*_pidvel < vel_limit_cur[3]) *_pidvel = vel_limit_cur[3];
}

void MPC_XPSAVT::getVel_limitted_Acc(double * _vel) {
	// limit range
	if (*_vel > velMax) *_vel = velMax;
	else if (*_vel < -velMax) *_vel = -velMax;
	// limit range(minimum
	if (fabs(*_vel) < 0.1) *_vel = 0;
	vel_limit_cur[2] = velMax;
	vel_limit_cur[3] = -velMax;

	if (n_impRsp_Acc == 0) return;
	// limit by acc
	double acc_sum = 0;
	for (int i = 4; i < n_impRsp_Acc; i++)
		acc_sum += impRsp_Acc[i] * u[i - 4];

	vel_limit_cur[0] = (Acc_limit - acc_sum) / impRsp_Acc[3];
	vel_limit_cur[1] = (-Acc_limit - acc_sum) / impRsp_Acc[3];

	// limit range from Acc
	if (*_vel > vel_limit_cur[0]) *_vel = vel_limit_cur[0];
	else if (*_vel < vel_limit_cur[1]) *_vel = vel_limit_cur[1];


	set_pidvel_limit(vel_limit_cur[0] - *_vel, vel_limit_cur[1] - *_vel);
}

void MPC_XPSAVT::getVel_limitted(double * _vel) {
	// limit dynamic range
	double dv = *_vel - u[predHrz - 1];
	if (abs(dv) > dvelMax) {
		if (dv < 0)
			dv = -dvelMax;
		else
			dv = dvelMax;
		*_vel = u[predHrz - 1] + dv;
	}
	// limit range
	if (*_vel > velMax) *_vel = velMax;
	else if (*_vel < -velMax) *_vel = -velMax;
}

void MPC_XPSAVT::getVel_limitted_fromCur(double * _vel, double _curvel) {
	// limit dynamic range
	double upper = _curvel + dvelMax;
	double lower = _curvel - dvelMax;
	// limit range (from current)
	if (*_vel > upper) *_vel = upper;
	else if (*_vel < lower) *_vel = lower;

	// limit range
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

void MPC_XPSAVT::Init(int _cntrHrz, double _wL1, double _wL2, double _velMax, double _dvelMax, int n, double * _impRsp) {
	updateParameters(_cntrHrz, _wL1, _wL2, _velMax, _dvelMax);
	updateImpRsp(n, _impRsp);
	generateMat();
	isInit = false;
}

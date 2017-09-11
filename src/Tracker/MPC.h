#ifndef _MPCXPSAVT_H
#define _MPCXPSAVT_H

#include "Base/base.h"
#include "mkl_lapacke.h"
#include "mkl_cblas.h"
#include "FeedbackControlMode.h"
#include "PIDcontroller.h"



class MPC_XPSAVT
{
public:
	MPC_XPSAVT(void);
	~MPC_XPSAVT(void);
	FeedbackControlMode controlmanager;
	PIDcontroller PIDCntr;
	void setVelMax(double _v);
	void setWeightL1(double _w);
	void setWeightL2(double _w);
	void setCntrHrz(int _steps);
	double getVelMax(void);
	double getWeightL1(void);
	double getWeightL2(void);
	int getCntrHrz(void);
	int getPredHrz(void);
	double getCurVel(void);
	double getCurPos(void);
	double * getP_precommitted(void);
	double * getVelHistory(void);
	double * getStpResp(void);
	double * getImpResp(void);
	void terminate(void);
	bool isInitiated(void);
	void Init(int _cntrHrz, double _wL1, double _wL2, double _velMax, double _dvelMax, int n, double * _impRsp);
	void updateImpRsp(int n, double * _impRsp);
	void updateImpRsp_Acc(int n, double * _impRsp_Acc, double _Acc_max);
	void updateParameters(int _cntrHrz, double _wL1, double _wL2, double _velMax, double _dvelMax);
	void step_weight(int n, double * p, double _wL1, double _wL2, double * nextPos, double * nextVel);
	double step_PID(double curpos, double desiredpos);
	void step(int n, double * p, double * nextPos, double * nextVel);
	void replaceNextInput(double _v);
	bool InitPos(double _initPos);
	void set_pidvel_limit(double _vmaxpid_upper, double _vmaxpid_lower);
	void set_pidvel_limit_max(void);
	double Acc_limit;
	double Acc_limit_set;

	double * impRsp_ALL;
	int impRsp_ALL_n;
	int impRsp_ALL_stride;

	double velMax;
	double dvelMax;


private:
	bool isInit;
	unsigned int predHrz;
	unsigned int cntrHrz;
	double weight_L1; // eye
	double weight_L2; // (-1 1)
	double * impRsp;
	double * impRsp_Acc;
	double vel_limit_cur[4]; // current maximum velocity, minimum velocity, remained to maximum velocity, remained to minimum velocity
	int n_impRsp_Acc;
	double * stpRsp;
	double * A; // include A and L1 and L2 (all)
	double * B;
	double * u; // history of input
	double * p_precommitted;
	void clear(void);
	void setPredHrz(int _steps);
	void generateMat(void); // before the calculation
	void setL1Mat(double _wL1);
	void setL2Mat(double _wL2);
	void copyDesiredPos(int n, double * _desiredPos);// before MPC calculation
	void update_p_precommoitted(double vel); /// after MPC calulation
	void update_p_precommoitted2(double vel); /// after MPC calulation
	void limit_PIDinput(double * _pidvel);
	void getVel_limitted_Acc(double * _vel);
	void getVel_limitted(double * _vel);
	void getVel_limitted_fromCur(double * _vel, double _curvel);
	void MPC(void);
};



#endif

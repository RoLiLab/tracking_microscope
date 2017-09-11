#ifndef _Piezo_NI_H
#define _Piezo_NI_H

#include "AnalogSignal.h"
#include "DSP/z_matcher.h"

class Piezo_NI
{
public:
	// basic functions
	Piezo_NI(void);
	Piezo_NI(double _freq, float64 amplitude, float64 center, int _layersPerScan);
	~Piezo_NI(void);

	AnalogSignal signalbase;// signal base + center
	AnalogSignal signal;
	AnalogSignal signal_archive;

	uInt64 totalupdatedSample;
	uInt64 offset_sampleupdatinglocation;
	uInt64 steps_update;
	uInt64 steps_per_sweep;
	uInt64 samples_per_step;
	uInt64 samples_offset;
	float64 center_um;
	float64 step_um;
	float64 step_um_gain2;
	float64 step_um_gain;
	float64 voltage_per_um;
	float64 sweepDirection;
	int update_location;
	int updateReadyOffset;
	bool isCenterUpdated;
	double updateCenter_um(double _center_um);


	// image process (FFT GPU module)
	int n_ref; // INPUT: The number of later in reference volume
	int id_ref_target; // INPUT: the target layer in reference volume
	double d_ref; // distance between layers in reference volume
	double z_target; // target position in reference volume
	int id_result; // the result index of the matched image in reference volume (return value from image processing)
	double z_result; // the result position from the result index
	double delta_z_res; // the difference between the z_result and z_target
	double delta_z_exp; // the expected differnce in position between the z_result and z_target
	int n_img; // a layer count per sweep
	int id_img; // image index (= frmNo % n_img)
	float cor_max;
	bool cor_max_updateEnable;
	int id_center; // the center location per sweep (= n_img/2)
	double d_img; // distance between layers in image stack
	double delta_z;// final result that need to adjust the center_um (+= delta_z_res - delta_z_exp)
	bool isdetaReset;// final result that need to adjust the center_um (+= delta_z_res - delta_z_exp)
	void set_reference(int _n_ref, double _d_ref);
	void set_reference_target(int _n_target);
	void set_image(int _n_img, double _d_img);
	float get_imagelocation_z_um(uint16 * _img);
	double get_adjust_z_um(int frmNo, uint16 * _img);
	void adjust_center_um(void);
	int frmNo_to_frmNoDouble(int);
	int frmNo_to_Index(int);
	bool isready_tracking;
	double dz_result_archive[1200]; // the result position from the result index
	double z_result_archive[1200]; // the result position from the result index
	int id_result_archive[1200]; // the result position from the result index
	void update_z_result(int frmNo);
	ZMatcher zmatcher_gpu;
	int id_lower_limit;
	int id_upper_limit;




	void getSignal_sawteeth(int _steps_per_sweep, float64 _amplitude_um, float64 _center_um);
	void getSignal_sawteeth(void);
	void getSignal_sawteeth_init(int _steps_per_sweep, float64 _step_um, float64 _center_um);
	void getSignal_sawteeth_update(void);
	void getSignal_sawteeth_update(float64 _center_um);
	void copy_toArchive(void);
	double getValue(uint64 FrmNo);
	bool isUpdateReady(uint64 curpos, uint64 targetpos);
	bool isUpdateReady(int sweepcount);
	void getSignal_half(int n);
	void getSignal_stop(void);

	void getSignalbase_triangle(void);
	void getSignal_triangle(int _steps_per_sweep, float64 _amplitude_um, float64 _center_um);

	double freq;
	int layersPerScan;
	uInt64 cycle_us;
	float64 minVolt_piezo;
	float64 maxVolt_piezo;
	void update(double _freq, float64 _minVolt_piezo, float64 _maxVolt_piezo, int _layersPerScan);
	void getSignal(double _freq, float64 amplitude, float64 center, int _layersPerScan);
	void getSignal(void);
private:
	void updatefreq(double _freq, int _layersPerScan);
};
#endif

#include "Base/base.h"
#include "Piezo_NI.h"

Piezo_NI::Piezo_NI(void)
{
	//float64 amplitude = 10;
	//float64 center = 5.0;
	//double _freq = 90.0; // layer freq
	//int _layerPerScan = 58; // scan freq = layer freq / layerPerScan
	//getSignal(_freq, amplitude, center, _layerPerScan);
	steps_update = 1000*50;
	int _steps_per_sweep = 50;
	float64 _center_um = 200;
	float64 _step_um = 1;
	voltage_per_um = 10.0 / 400.0; 
	sweepDirection = 1.0;
	getSignal_sawteeth(_steps_per_sweep, _center_um, _step_um);
	signal_archive.resetSignal(1200, 0);
	update_location = 0;
	step_um_gain = 0;
	step_um_gain2 = 0;
	updateReadyOffset = 10;
	isCenterUpdated = true;
	isdetaReset = false;
	id_lower_limit = 5;
	id_upper_limit = 45;
	// ---------

	// image process (FFT GPU module)
	isready_tracking = false;
	n_ref = 0; // INPUT: The number of later in reference volume
	id_ref_target = 0; // INPUT: the target layer in reference volume
	d_ref = 0; // distance between layers in reference volume
	z_target = 0; // target position in reference volume
	id_result = 0; // the result index of the matched image in reference volume (return value from image processing)
	z_result = 0; // the result position from the result index
	delta_z_res = 0; // the difference between the z_result and z_target
	delta_z_exp = 0; // the expected differnce in position between the z_result and z_target
	n_img = 0; // a layer count per sweep
	id_img = 0; // image index (= frmNo % n_img)
	d_img = 0; // distance between layers in image stack
	id_center = 25; // the center location per sweep (= n_img/2)
	delta_z = 0;// final result that need to adjust the center_um (+= delta_z_res - delta_z_exp)
	cor_max_updateEnable = true;
	
}

Piezo_NI::Piezo_NI(double _freq, float64 amplitude, float64 center, int _layersPerScan)
{
	//getSignal(_freq, amplitude, center, _layersPerScan);
}

Piezo_NI::~Piezo_NI()
{
}
// --------------------------- buffer approach -------------------------------------//
void Piezo_NI::getSignal_sawteeth(void) {
	updateCenter_um(center_um);
	float64 p0 = (center_um - step_um *step_um_gain*step_um_gain2*(float64)steps_per_sweep / 2) * voltage_per_um;
	float64 dp = (step_um*step_um_gain*step_um_gain2)*voltage_per_um;
	
	for (int i = 0; i < steps_per_sweep; i++) {
		float64 v = p0 + i*dp;
		for (int j = 0; j < samples_per_step; j++)
			signalbase.data[i*samples_per_step + j] = v;
	}

	//signalbase.resetSignal_value(center_um*voltage_per_um);
	//for (int i = 0; i < 1000; i++)
	//	signalbase.data[i] = 8.0;
	//for (int i = steps_update - 1000; i < steps_update; i++)
	//	signalbase.data[i] = 2.0;
}


void Piezo_NI::getSignal_sawteeth(int _steps_per_sweep, float64 _step_um, float64 _center_um) {
	if (steps_per_sweep != _steps_per_sweep || (signal.n == 0)) {
		steps_per_sweep = _steps_per_sweep;
		samples_per_step = steps_update / steps_per_sweep;
		steps_per_sweep = _steps_per_sweep;
		signal.resetSignal(steps_update, 0.0);
		signalbase.resetSignal(steps_update, 0.0);
	}
	id_center = _steps_per_sweep / 2;
	center_um = _center_um;
	step_um = _step_um;
	getSignal_sawteeth();
}


void Piezo_NI::getSignal_sawteeth_init(int _steps_per_sweep, float64 _step_um, float64 _center_um) {
	getSignal_sawteeth(_steps_per_sweep, _step_um, _center_um);
	signal.resetSignal(steps_update, 0.0);
	totalupdatedSample = 0;
	offset_sampleupdatinglocation = signalbase.n + 1;
	//signal.generateAnalogSignal(&signalbase, 0, steps_update, signalbase.data[0]);
	copy_toArchive();
	totalupdatedSample += steps_per_sweep;
	//isCenterUpdated = false;
}
void Piezo_NI::getSignal_sawteeth_update(void) {
	getSignal_sawteeth();
	signal.resetSignal(steps_update, 0.0);
	//signal.generateAnalogSignal(&signalbase, 0, steps_update, signalbase.data[0]);
	copy_toArchive();
	totalupdatedSample += steps_per_sweep;
	//isCenterUpdated = false;
}

void Piezo_NI::getSignal_sawteeth_update(float64 _center_um) {
	center_um = _center_um + delta_z;
	getSignal_sawteeth_update();
	delta_z = 0;
}

void Piezo_NI::copy_toArchive(void) {
	for (int i = 0; i < steps_per_sweep; i++)
		signal_archive.data[(totalupdatedSample + i) % signal_archive.n] = signalbase.data[i*samples_per_step] / voltage_per_um;
}

void Piezo_NI::getSignal_stop() {
	if (step_um_gain2 == 0)
		step_um_gain2 = 1;
	else
		step_um_gain2 = 0; 
}
double Piezo_NI::getValue(uint64 FrmNo) {
	int frmNoEPI = frmNo_to_Index((int)FrmNo);
	return (double)signal_archive.data[frmNoEPI];
}

bool Piezo_NI::isUpdateReady(uint64 curpos, uint64 targetpos) {
	update_location = (targetpos - curpos) / samples_per_step;
	return (update_location < updateReadyOffset);
}

bool Piezo_NI::isUpdateReady(int sweepcount) {
	return (sweepcount >= steps_per_sweep - updateReadyOffset);
}


double Piezo_NI::updateCenter_um(double _center_um) {
	center_um = _center_um;
	double dz0 = step_um *step_um_gain*step_um_gain2* (float64)steps_per_sweep / 2;
	if (center_um + dz0 > 400) center_um = 400 - dz0;
	if (center_um - dz0 < 0) center_um = dz0;
	return center_um;
}
// --------------------------- END: buffer approach -------------------------------------//

// --------------------------- FFT matching adjust -------------------------------------//
void Piezo_NI::set_reference(int _n_ref, double _d_ref) {
	n_ref = _n_ref;
	d_ref = _d_ref;
}
void Piezo_NI::set_reference_target(int _n_target) {
	id_ref_target = _n_target;
	z_target = id_ref_target * d_ref;
}
void Piezo_NI::set_image(int _n_img, double _d_img) {
	n_img = _n_img; // = steps_per_sweep
	d_img = _d_img; // = step_um
}
float Piezo_NI::get_imagelocation_z_um(uint16 * _img) {	
	float cor = 0;
	id_result = zmatcher_gpu.match_z(_img, &cor);
	z_result = id_result * d_ref;
	delta_z_res = z_result - z_target;
	return cor;
}
int Piezo_NI::frmNo_to_frmNoDouble(int frmNo) {
	return (frmNo+200)/2;
}
int Piezo_NI::frmNo_to_Index(int frmNo) {
	return (frmNo_to_frmNoDouble(frmNo) % 1200);
}

void Piezo_NI::update_z_result(int frmNo) {
	int _id = frmNo_to_Index(frmNo);
	z_result_archive[_id] = delta_z_res;
	dz_result_archive[_id] = delta_z;
	id_result_archive[_id] = id_result;
}

double Piezo_NI::get_adjust_z_um(int frmNo, uint16 * _img) {
	// get a position in refrence stack
	int frmNoEPI = frmNo_to_frmNoDouble(frmNo);
	float cor = 0;
	if (zmatcher_gpu.enable_simulation)
		cor = get_imagelocation_z_um(zmatcher_gpu.buffer_simulation[frmNo%zmatcher_gpu.n_simulation]); // return value: delta_z_res
	else
		cor = get_imagelocation_z_um(_img); // return value: delta_z_res

	// get a position in real sweep

	if ((id_center >= id_lower_limit) && (id_center <= id_upper_limit)) {
		id_img = frmNoEPI  % steps_per_sweep;
		delta_z_exp = ((double)id_img - (double)id_center) * (double)step_um; 
		// get a difference
		if (isdetaReset == true) {
			cor_max = 0;
			isdetaReset = false;
		}
		if ((cor > cor_max) || cor_max_updateEnable){ //
			cor_max = cor;
			delta_z = -(delta_z_res - delta_z_exp);
		}
	}
	else {
		delta_z_exp = 0;
	}
	update_z_result(frmNo);	
	return delta_z;
}
void Piezo_NI::adjust_center_um(void){
	// update center_um
	updateCenter_um((double)center_um + delta_z);
}


// --------------------------- END: FFT matching adjust -------------------------------------//



void Piezo_NI::getSignal(double _freq, float64 amplitude, float64 center, int _layersPerScan) 
{
	update(_freq, amplitude, center, _layersPerScan);
	getSignal();
}

void Piezo_NI::update(double _freq, float64 amplitude, float64 center, int _layersPerScan)
{
	updatefreq(_freq, _layersPerScan);
	minVolt_piezo = center - amplitude/2;
	maxVolt_piezo = center + amplitude/2;
}

void Piezo_NI::updatefreq(double _freq, int _layersPerScan)
{
	freq = _freq;
	layersPerScan = _layersPerScan;
	cycle_us = (uInt64)(1000000.0/freq*(double)_layersPerScan);
}


void Piezo_NI::getSignal(void)
{
	signal.generateAnalogSignal(AnalogSignal::AO_Sawtooth, cycle_us*10, minVolt_piezo, maxVolt_piezo);
}

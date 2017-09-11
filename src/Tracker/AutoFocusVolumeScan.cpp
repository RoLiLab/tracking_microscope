#include "Base/base.h"
#include "Tracker/AutoFocusVolumeScan.h"

AutoFocusVolumeScan::AutoFocusVolumeScan(void)
{
	mode = 1; // adaptive - sawtooth
	mask = NULL;
	sawthooth_buffer = NULL;
	samestep_size = 1;
	dz = 0;
	histlen = 250 * 4; //250 * 5
	slidewinsize = 10;
	updateparameters(2.0, 300, 0.5, 0.55);
	updateparameters_imgSetup(1800, 1800);
	lockdown_steps = 0;
	lockdown_stepsset = 10;
	lockdown_minchange = 0.0;
	lockdown_minchangeset = 0.1;
	z = vector<double>(histlen);
	mu = vector<double>(histlen);
	mu_mva = vector<double>(histlen);
	z_real = vector<double>(histlen);
	imgdata = (FLRImageData *)malloc(histlen * sizeof(FLRImageData));
	imgdata_cur = NULL;
	FLRImageData * imgData;
	FLRImageData * imgData_cur;

	imgMeanIntensity_DIFF[0] = 0;
	imgMeanIntensity_DIFF[1] = 0;

	for (int i = 0; i < histlen; i++){
		z[i] = 200;
		mu[i] = 0;
		mu_mva[i] = 0;
		z_real[i] = 200;
	}
	zmax = 400.0;
	zmin = 0.0;
	z_top = 0.0;
	z_bottom = 0.0;
	isfocused_cur = false;
	isfocused_prev = false;
	piezo_inputVolts = 5.0;
	mu_max = 0;
	mu_min = 0;

	vmean_cur = 0;
	vmean_max = 0;
	vmean_max_last = 200;
	vmean_min = 0;
	vmean_min_last = 100;
	z_vmean_max = 0;
	z_vmean_max_offset = 0;
	z_vmean_max_index = 0;
	z_vmean_max_last = 200;
	z_init = 200;
}

void AutoFocusVolumeScan::updateparameters(double _dzset, uint16 _threshold, double _mu_low, double _mu_high)
{
	dzset = _dzset;
	if (dz == 0)
		dz = 0;
	else
		dz = dz / fabs(dz) * dzset;

	threshold = _threshold;
	lockdown_minchangeset = _mu_high - _mu_low;
	mu_low = _mu_low;
	if (_mu_low < _mu_high)
		mu_high = _mu_high;
	else
		mu_high = mu_low + 0.05;
	mu_threshold = mu_low;

	if (lockdown_stepsset > 0) {
		if (sawthooth_buffer) free(sawthooth_buffer);
		sawthooth_buffer = (double *)malloc(lockdown_stepsset * sizeof(double));
		for (int i = 0; i < lockdown_stepsset; i++)
			sawthooth_buffer[i] = z_init + dzset*i;
	}
	lockdown_steps = 0;
}


void AutoFocusVolumeScan::updateparameters_fixed(double _dzset, uint16 _threshold, double _mu_low, double _mu_high)
{
	dzset = _dzset;
	double dz0 = _dzset*lockdown_stepsset / 2;
	if (lockdown_stepsset > 0) {
		if (sawthooth_buffer) free(sawthooth_buffer);
		sawthooth_buffer = (double *)malloc(lockdown_stepsset * sizeof(double));
		for (int i = 0; i < lockdown_stepsset; i++)
			sawthooth_buffer[i] = z_init - dz0 + dzset*i;
	}
	lockdown_steps = 0;
}

void AutoFocusVolumeScan::updateparameters_imgSetup(int _w, int _h){
	imgsize.width = _w;
	imgsize.height = _h;

	if (mask)
	free(mask);
	mask = (unsigned char *)malloc(imgsize.width*imgsize.height* sizeof(unsigned char));

}

AutoFocusVolumeScan::~AutoFocusVolumeScan(void)
{
	if (sawthooth_buffer) free(sawthooth_buffer);
	if (imgdata) free(imgdata);
}

double AutoFocusVolumeScan::compute_mu(uint16 * pImg, int _FrmNo) {

	if (pImg == NULL) return 0.0;

	int idx = _FrmNo%histlen;
	imgdata[idx].FrmNo = _FrmNo;

	IppStatus error = ippiCompareC_16u_C1R((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp16u)threshold, (Ipp8u*)mask, imgsize.width, imgsize, ippCmpGreaterEq);
	error = ippiMean_8u_C1R((Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].filled);
	error = ippiMean_16u_C1MR((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].intensity_mean);
	if (imgdata[idx].intensity_mean > 0) {
		ippiMax_16u_C1R((Ipp16u*)pImg, imgsize.width*sizeof(uint16), imgsize, (Ipp16u*)(&imgdata[idx].intensity_max));
		mu_cur = (1.0 - (imgdata[idx].intensity_mean / (double)imgdata[idx].intensity_max));
	}
	imgdata_cur = &imgdata[idx];

	mu.push_back(mu_cur); mu.erase(mu.begin());

	//imgdata
	//mu_max = *max_element(mu.begin(), mu.end());
	//mu_min = *min_element(mu.begin(), mu.end());

	double mu_sum = 0;
	for (int i = 0; i < slidewinsize; i++) {
		if (mu[mu.size() - 1] == 0) {
			mu_sum = 0;
			break;
		}
		else
			mu_sum += mu[mu.size() - 1 - i];
	}
	mu_cur = mu_sum / slidewinsize;
	mu_mva.push_back(mu_cur); mu_mva.erase(mu_mva.begin());
	mu_max = *max_element(mu_mva.begin(), mu_mva.end());
	mu_min = *min_element(mu_mva.begin(), mu_mva.end());
	return mu_cur;
}

FLRImageData * AutoFocusVolumeScan::getimgdatafromIdx(int _FrmNo) {
	FLRImageData * curdata = NULL;
	int idx = _FrmNo%histlen;
	if (imgdata[idx].FrmNo == _FrmNo)
		curdata = &imgdata[idx];
	return curdata;
}
double AutoFocusVolumeScan::compute_mean_GS3(uint16 * pImg, int _FrmNo) {

	if (pImg == NULL) return 0.0;

	int idx = _FrmNo%histlen;
	imgdata[idx].FrmNo = _FrmNo;
	int len = imgsize.width*imgsize.height;
	unsigned char * p = (unsigned char *)pImg;
	int count = 0;
	double int_sum = 0;
	double int_max = 0;
	for (int i = 0; i < len; i = i + 2) {

		unsigned char * b0 = p;
		unsigned char * b1 = p + 1;
		unsigned char * b2 = p + 2;
		uint16 v1 = ((uint16)(*b0) << 4) | (*b1 & 0x0F);
		uint16 v2 = ((uint16)(*b2) << 4) | (*b1 & 0xF0);

		p = p + 3;
		if (v1 > threshold) {
			count++;
			int_sum += (double)v1;
		}
		if (v1 > int_max) int_max = v1;
		if (v2 > threshold) {
			count++;
			int_sum += (double)v2;
		}
		if (v2 > int_max) int_max = v2;
	}
	imgdata[idx].intensity_max = int_max;
	imgdata[idx].filled = (double)count/len;
	imgdata[idx].intensity_mean = int_sum/len;
	//IppStatus error = ippiCompareC_16u_C1R((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp16u)threshold, (Ipp8u*)mask, imgsize.width, imgsize, ippCmpGreaterEq);
	//error = ippiMean_8u_C1R((Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].filled); imgdata[idx].filled /= 255.0;
	//error = ippiMean_16u_C1MR((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].intensity_mean);

	imgdata_cur = &imgdata[idx];

	vmean_cur = imgdata[idx].intensity_mean;
	if (vmean_cur > vmean_max) {
		vmean_max = vmean_cur;
		z_vmean_max = z_real[histlen - 1] + z_vmean_max_offset;
		z_vmean_max_index = idx;
	}
	else if (vmean_cur < vmean_min && vmean_cur > 0) {
		vmean_min = vmean_cur;
	}
	mu.push_back(vmean_cur); mu.erase(mu.begin());
	return vmean_cur;
}
double AutoFocusVolumeScan::compute_mean_GS3u10(uint16 * pImg, int _FrmNo) {

	if (pImg == NULL) return 0.0;

	int idx = _FrmNo%histlen;
	imgdata[idx].FrmNo = _FrmNo;
	int len = imgsize.width*imgsize.height;
	uint16 * p = (uint16 *)pImg;
	int count = 0;
	double int_sum = 0;
	double int_sum_all = 0;
	double int_max = 0;
	p = p + 10;
	for (int i = 10; i < len; i = i + 1) {
		double v1 = (double)((*p) >> 6);
		if (v1 > threshold) {
			count++;
			int_sum += (double)v1;
		}
		if (v1 > int_max) int_max = v1;
		int_sum_all += v1;
		p = p + 1;
	}
	int_sum_all /= len;

	imgdata[idx].intensity_max = int_max;
	imgdata[idx].filled = (double)count / len;
	if (count > 0)
		imgdata[idx].intensity_mean = int_sum / count;
	else
		imgdata[idx].intensity_mean = 0;


	imgMeanIntensity_DIFF[_FrmNo % 2] = imgMeanIntensity_DIFF[_FrmNo % 2] * (74 / 75) + int_sum_all;
	imgMeanIntensity_DIFF_ratio = (imgMeanIntensity_DIFF[0] - imgMeanIntensity_DIFF[1]) / imgMeanIntensity_DIFF[1] * 100;
	//IppStatus error = ippiCompareC_16u_C1R((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp16u)threshold, (Ipp8u*)mask, imgsize.width, imgsize, ippCmpGreaterEq);
	//error = ippiMean_8u_C1R((Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].filled); imgdata[idx].filled /= 255.0;
	//error = ippiMean_16u_C1MR((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].intensity_mean);

	imgdata_cur = &imgdata[idx];

	vmean_cur = imgdata[idx].intensity_mean;
	if (vmean_cur > vmean_max) {
		vmean_max = vmean_cur;
		z_vmean_max = z_real[histlen - 1] + z_vmean_max_offset;
		z_vmean_max_index = idx;
	}
	else if (vmean_cur < vmean_min && vmean_cur > 0) {
		vmean_min = vmean_cur;
	}
	mu.push_back(vmean_cur); mu.erase(mu.begin());
	return vmean_cur;
}
double AutoFocusVolumeScan::compute_mean(uint16 * pImg, int _FrmNo) {

	if (pImg == NULL) return 0.0;

	int idx = _FrmNo%histlen;
	imgdata[idx].FrmNo = _FrmNo;

	IppStatus error = ippiCompareC_16u_C1R((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp16u)threshold, (Ipp8u*)mask, imgsize.width, imgsize, ippCmpGreaterEq);
	error = ippiMean_8u_C1R((Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].filled); imgdata[idx].filled /= 255.0;
	error = ippiMean_16u_C1MR((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp8u*)mask, imgsize.width, imgsize, &imgdata[idx].intensity_mean);
	//error = ippiMax_16u_C1R((Ipp16u*)pImg, imgsize.width*sizeof(uint16), imgsize, (Ipp16u*)(&imgdata[idx].intensity_max));
	imgdata_cur = &imgdata[idx];


	imgMeanIntensity_DIFF[_FrmNo % 2] = imgMeanIntensity_DIFF[_FrmNo % 2] * (74 / 75) + imgdata[idx].intensity_mean;
	imgMeanIntensity_DIFF_ratio = (imgMeanIntensity_DIFF[0] - imgMeanIntensity_DIFF[1]) / imgMeanIntensity_DIFF[1] * 100;


	vmean_cur = imgdata[idx].intensity_mean;
	if (vmean_cur > vmean_max) {
		vmean_max = vmean_cur;
		z_vmean_max = z_real[histlen - 1] + z_vmean_max_offset;
		z_vmean_max_index = idx;
	}
	else if (vmean_cur < vmean_min && vmean_cur > 0) {
		vmean_min = vmean_cur;
	}
	mu.push_back(vmean_cur); mu.erase(mu.begin());
	return vmean_cur;




}
double AutoFocusVolumeScan::compute_mean_old(uint16 * pImg) {

	if (pImg == NULL) return 0.0;

	double v_mean = 0;
	uint16 v_max = 0;
	IppStatus error = ippiCompareC_16u_C1R((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp16u)threshold, (Ipp8u*)mask, imgsize.width, imgsize, ippCmpGreaterEq);
	error = ippiMean_16u_C1MR((Ipp16u*)pImg, imgsize.width*sizeof(uint16), (Ipp8u*)mask, imgsize.width, imgsize, &v_mean);

	vmean_cur = v_mean;
	if (vmean_cur > vmean_max) {
		vmean_max = vmean_cur;
		z_vmean_max = z_real[histlen - 1];
	}
	else if (vmean_cur < vmean_min && vmean_cur > 0) {
		vmean_min = vmean_cur;
	}
	mu.push_back(vmean_cur); mu.erase(mu.begin());
	return vmean_cur;
}

double AutoFocusVolumeScan::update_step(void) {
	double voltage = 0;
	switch (mode) {
	case 0:
		voltage = update_step_sweep();
		break;
	case 1:
		voltage = update_step_sweep_1D();
		break;
	case 2:
		voltage = update_step_sweep_1D_fixed();
		break;
	case 3:
		voltage = update_step_sweep_fixed();
		break;
	default:
		voltage = 0;
		break;
	}
	return voltage;
}
double AutoFocusVolumeScan::update_step_sweep(void)
{
	// step 2 - compute z position
	if (0) {
	//if (vmean_cur == 0) {
		dz = dzset; lockdown_steps = lockdown_stepsset;// reverse direction and lock down set
		z_init = z[histlen - 1];
		vmean_max = 0; z_vmean_max = z_init;
		vmean_min = 70000;
	}
	else {
		if (lockdown_steps == 0) {

			vmean_max_last = vmean_max;
			vmean_min_last = vmean_min;
			if (vmean_min_last >= vmean_max_last)
				vmean_min_last = vmean_max_last + 1;

			z_vmean_max_last = z_vmean_max;
			vmean_min = 70000;
			vmean_max = 100;
			//if (0){
			if (dz >= 0){
				dz = -dzset;
				z_init = z_vmean_max + (double)lockdown_stepsset* dzset /2;
				if (z_init > zmax) z_init = zmax;
				z_vmean_max = z_init;
				lockdown_steps = lockdown_stepsset;
			}
			else {
				dz = dzset;
				z_init = z_vmean_max - (double)lockdown_stepsset* dzset / 2;
				if (z_init < zmin) z_init = zmin;
				z_vmean_max = z_init;
				lockdown_steps = lockdown_stepsset;
			}
		}
		else {
			lockdown_steps--;
		}
	}

	// step 3 - limit (at the limit, reset)
	double z_cur = z_init + (lockdown_stepsset - lockdown_steps)*dz;
	if (z_cur >= zmax) {
		dz = -dzset; vmean_max = 0;
		z_init = zmax; z_vmean_max = z_init;
		lockdown_steps = lockdown_stepsset;
		z_cur = z_init;
	}
	else if (z_cur <= zmin) {
		dz = dzset; vmean_max = 0;
		z_init = zmin; z_vmean_max = z_init;
		lockdown_steps = lockdown_stepsset;
		z_cur = z_init;
	}

	// step 3 - update
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());
	//return z_cur;

	// step 4. convert um -> volt
	if (z_cur > 400) z_cur = 400;
	if (z_cur < 0) z_cur = 0;
	piezo_inputVolts = 10.0*(400 - z_cur) / 400.0;

	imgdata_cur->piezo_z[0] = z_cur;
	return piezo_inputVolts;

}

double AutoFocusVolumeScan::update_step_sweep_fixed(void)
{

	// step 2 - compute z position
	if (lockdown_steps == 0) {
		vmean_max_last = vmean_max;
		vmean_min_last = vmean_min;
		if (vmean_min_last >= vmean_max_last)
			vmean_min_last = vmean_max_last + 1;

		z_vmean_max_last = z_vmean_max;
		vmean_min = 70000;
		vmean_max = 100;
		//if (0){
		if (dz >= 0){
			dz = -dzset;
			z_init = z_vmean_max + (double)lockdown_stepsset* dzset / 2;
			//if (z_init > zmax) z_init = zmax;
			//z_vmean_max = z_init;
			lockdown_steps = lockdown_stepsset;
		}
		else {
			dz = dzset;
			z_init = z_vmean_max - (double)lockdown_stepsset* dzset / 2;
			//if (z_init < zmin) z_init = zmin;
			//z_vmean_max = z_init;
			lockdown_steps = lockdown_stepsset;
		}
	}
	else {
		lockdown_steps--;
	}

	// step 3 - limit (at the limit, reset)
	double z_cur = z_init + (lockdown_stepsset - lockdown_steps)*dz;
	if (z_cur >= zmax) {
		dz = -dzset; vmean_max = 0;
		z_init = zmax;
		z_vmean_max = z_init;
		lockdown_steps = lockdown_stepsset;
		z_cur = z_init;
	}
	else if (z_cur <= zmin) {
		dz = dzset; vmean_max = 0;
		z_init = zmin;
		z_vmean_max = z_init;
		lockdown_steps = lockdown_stepsset;
		z_cur = z_init;
	}

	// step 3 - update
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());
	//return z_cur;

	// step 4. convert um -> volt
	if (z_cur > 400) z_cur = 400;
	if (z_cur < 0) z_cur = 0;
	piezo_inputVolts = 10.0*(400 - z_cur) / 400.0;

	imgdata_cur->piezo_z[0] = z_cur;
	return piezo_inputVolts;
}

double AutoFocusVolumeScan::update_step_sweep_1D(void)
{
	double z_cur = z_init;
	if (sawthooth_buffer) {
		z_cur = sawthooth_buffer[lockdown_steps++];
		if ((z_cur >= 400) || (z_cur <= 0) || lockdown_steps == lockdown_stepsset) {
			z_vmean_max_last = z_vmean_max;
			vmean_min = 70000;
			vmean_max = 100;
			lockdown_steps = 0;
			int maxIndex = z_vmean_max_index%lockdown_stepsset;
			z_init = z_vmean_max - (dzset*lockdown_stepsset) / 2;
			//if ((z_vmean_max_index >= 0) && (z_vmean_max_index <= 7)) z_init = z_vmean_max - (dzset*lockdown_stepsset);
			//if ((z_vmean_max_index <= lockdown_stepsset - 0) && (z_vmean_max_index >= lockdown_stepsset - 7)) z_init = z_vmean_max;

			if (z_init < 0)
				z_init = 0;
			if (z_init > 400.0 - (dzset*lockdown_stepsset))
				z_init = 400.0 - (dzset*lockdown_stepsset);
			for (int i = 0; i < lockdown_stepsset; i++)
				sawthooth_buffer[i] = z_init + dzset*i;
		}
	}

	// step 3 - update
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());
	//return z_cur;

	// step 4. convert um -> volt
	if (z_cur > 400) z_cur = 400;
	if (z_cur < 0) z_cur = 0;
	piezo_inputVolts = 10.0*(400 - z_cur) / 400.0;

	imgdata_cur->piezo_z[0] = z_cur;
	return piezo_inputVolts;

}


double AutoFocusVolumeScan::update_step_sweep_1D_fixed(void)
{
	double z_cur = z_init;
	if (sawthooth_buffer) {
		z_cur = sawthooth_buffer[lockdown_steps++];
		if ((z_cur >= 400) || (z_cur <= 0) || lockdown_steps == lockdown_stepsset)
			lockdown_steps = 0;
	}

	// step 3 - update
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());
	//return z_cur;

	// step 4. convert um -> volt
	if (z_cur > 400) z_cur = 400;
	if (z_cur < 0) z_cur = 0;
	piezo_inputVolts = 10.0*(400 - z_cur) / 400.0;

	imgdata_cur->piezo_z[0] = z_cur;
	return piezo_inputVolts;

}

double AutoFocusVolumeScan::update_step_minchange(void)
{
	// step 1 - get mu

	// step 2 - compute z position
	if (mu_cur < 0.1) {
		dz = -dzset; lockdown_minchange = lockdown_minchangeset;// reverse direction and lock down set
	}
	else {
		if (lockdown_minchange == 0.0) {
			if (mu_cur < mu_threshold) {
				dz = -dz; lockdown_minchange = lockdown_minchangeset;// reverse direction and lock down set
			}
		}
		else {
			if (mu_cur > mu_threshold + lockdown_minchange) {
				lockdown_minchange = 0.0;//
			}
		}
	}

	// step 3 - limit (at the limit, reset)
	double z_cur = z[histlen - 1] + dz;
	if (z_cur > zmax) {
		dz = -dz; lockdown_minchange = lockdown_minchangeset;// reverse direction and lock down set
		z_cur = zmax;
	}
	else if (z_cur < zmin) {
		dz = -dz; lockdown_minchange = lockdown_minchangeset;// reverse direction and lock down set
		z_cur = zmin;
	}

	// step 3 - update
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());
	//return z_cur;

	// step 4. convert um -> volt
	piezo_inputVolts = 10.0*z_cur/400.0;
	return piezo_inputVolts;
}

double AutoFocusVolumeScan::update_steps(void)
{
	// step 1 - get mu
	double _dzcur = dz;
	// step 2 - compute z position
	if (mu_cur == 0) {
		dz = -dzset; lockdown_steps = lockdown_stepsset;// reverse direction and lock down set
	}
	else {
		if (lockdown_steps == 0) {
			if (mu_cur < mu_threshold) {
				dz = -_dzcur; lockdown_steps = lockdown_stepsset;// reverse direction and lock down set
			}
		}
		else {
			lockdown_steps--;
		}
	}

	// step 3 - limit (at the limit, reset)
	double z_cur = z[histlen - 1] + dz;
	if (z_cur > zmax) {
		dz = -_dzcur; lockdown_steps = lockdown_stepsset;// reverse direction and lock down set
		z_cur = zmax;
	}
	else if (z_cur < zmin) {
		dz = -_dzcur; lockdown_steps = lockdown_stepsset;// reverse direction and lock down set
		z_cur = zmin;
	}

	// step 3 - update
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());
	//return z_cur;

	// step 4. convert um -> volt
	piezo_inputVolts = 10.0*z_cur / 400.0;
	return piezo_inputVolts;

}

double AutoFocusVolumeScan::update_old(void)
{
	// step 1 - get mu

	// step 2 - compute z position
	isfocused_cur = (mu_cur > mu_threshold);
	if (mu_cur == 0) {
		dz = -dzset;
		mu_threshold = mu_high;
	}
	else {
		if (mu_cur < mu_threshold && mu_threshold == mu_low) {
			dz = -dz;
			mu_threshold = mu_high;
		}
		else if (mu_cur > mu_threshold && mu_threshold == mu_high){
			mu_threshold = mu_low;
		}
	}
	isfocused_cur = (mu_cur > mu_threshold);

	// step 3 - limit
	double z_cur = z[histlen - 1] + dz;
	if (z_cur > zmax) {
		z_cur = zmax;
		dz = -dz;
	}
	else if (z_cur < zmin) {
		dz = -dz;
		z_cur = zmin;
	}

	// step 3 - update
	isfocused_prev = isfocused_cur;
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());
	//return z_cur;

	// step 4. convert um -> volt
	piezo_inputVolts = 10.0*z_cur / 400.0;
	return piezo_inputVolts;

}

double AutoFocusVolumeScan::up(void)
{
	// step 3 - limit
	double z_cur = z[histlen - 1] + dzset;
	if (z_cur > zmax) {
		z_cur = zmax;
	}
	else if (z_cur < zmin) {
		z_cur = zmin;
	}
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());

	// step 4. convert um -> volt
	piezo_inputVolts = 10.0*z_cur / 400.0;
	return piezo_inputVolts;
}


double AutoFocusVolumeScan::down(void)
{
	// step 3 - limit
	double z_cur = z[histlen - 1] - dzset;
	if (z_cur > zmax) {
		z_cur = zmax;
	}
	else if (z_cur < zmin) {
		z_cur = zmin;
	}
	z.push_back(z_cur);	z.erase(z.begin());
	z_top = *max_element(z.begin(), z.end());
	z_bottom = *min_element(z.begin(), z.end());

	// step 4. convert um -> volt
	piezo_inputVolts = 10.0*z_cur / 400.0;
	return piezo_inputVolts;
}

/*

vector<AutoFocusVolumeScan> createScanTable(double scanCenter, double scanRange, int scanStep, double piezodir, int scanImgsPerLayer) {
	vector<AutoFocusVolumeScan> piezoInputTable(0);
	int Piezo_scanNumLayer = (int)(scanRange/(double)scanStep) + 1;
	for (int i = 0; i < Piezo_scanNumLayer; i++) {
		int PiezoPosUm = (int)(scanCenter + (piezodir) * ( (scanStep) * i - (scanRange/2) ));
		if (0 <= PiezoPosUm && PiezoPosUm <= 400) {
			AutoFocusVolumeScan temp_data(PiezoPosUm, 0);
			for (int j = 0; j < scanImgsPerLayer; j++)
				piezoInputTable.push_back(temp_data);
		}
	}
	return piezoInputTable;
}



void AutoFocusVolumeScan::Initialization(void) {
	m_ImageCompleteFlag[0] = false;
	m_ImageCompleteFlag[1] = false;
	m_ImageCompleteFlag[2] = false;
	m_VolumeScanCompleteFlag = false;
	m_FocusValue[0] = 0;
	m_FocusValue[1] = 0;
	m_FocusValue[2] = 0;
	m_zPos[1] = 200;
	m_zPos[0] = m_zPos[1] - m_dz;
	m_zPos[2] = m_zPos[1] + m_dz;
	m_ScanSequence[0] = Middle;
	m_ScanSequence[1] = Lower;
	m_ScanSequence[2] = Upper;
	m_CurrentPlane = 0;
	m_NextFocalPlane = 1;
}

void AutoFocusVolumeScan::Initialization(double zPos) {
	m_ImageCompleteFlag[0] = false;
	m_ImageCompleteFlag[1] = false;
	m_ImageCompleteFlag[2] = false;
	m_VolumeScanCompleteFlag = false;
	m_FocusValue[0] = 0;
	m_FocusValue[1] = 0;
	m_FocusValue[2] = 0;
	m_zPos[1] = zPos;
	m_zPos[0] = m_zPos[1] - m_dz;
	m_zPos[2] = m_zPos[1] + m_dz;
}

bool AutoFocusVolumeScan::isCompleteVolumeScan(void){ return m_VolumeScanCompleteFlag;} // return volume scan completion flag
void AutoFocusVolumeScan::setZMax(double zMax){ m_zMax = zMax;} // set z-maximum position
double AutoFocusVolumeScan::getZMax(void){ return m_zMax;} // get z-maximum position
void AutoFocusVolumeScan::setZMin(double zMin){ m_zMin = zMin;} // set z-minimum position
double AutoFocusVolumeScan::getZMin(void){ return m_zMin;} // get z-minimum position
void AutoFocusVolumeScan::setdz(double dz){ m_dz = dz;} // set dz value
double AutoFocusVolumeScan::getdz(void){ return m_dz;} // get dz value
void AutoFocusVolumeScan::setdzMax(double dzMax){ m_dzMax = dzMax;} // set dz max value
double AutoFocusVolumeScan::getdzMax(void){return m_dzMax;} // get dz max value
void AutoFocusVolumeScan::setdzMin(double dzMin){ m_dzMin = dzMin;} // set dz min value
double AutoFocusVolumeScan::getdzMin(void){ return m_dzMin;} // get dz min value
void AutoFocusVolumeScan::setScanSequence(int * Scan) { m_ScanSequence[0] = Scan[0];m_ScanSequence[1] = Scan[1];m_ScanSequence[2] = Scan[2];} // set the scan sequence (layer)
int * AutoFocusVolumeScan::getScanSequence(void) { return m_ScanSequence;} // get the scan sequence (layer)
void AutoFocusVolumeScan::setROI(cv::Rect ROI) { m_ROI = ROI;} // set a region of Interest for an image
cv::Rect AutoFocusVolumeScan::getROI(void) { return m_ROI;} // get a region of Interest for an image
void AutoFocusVolumeScan::setCurrentScanLayer(int layer) { m_CurrentPlane = layer;} // get the current plane layer
void AutoFocusVolumeScan::updateCurrentScanLayer (void){m_CurrentPlane = m_NextFocalPlane;}; // setCurrent plane layer
int AutoFocusVolumeScan::getCurrentScanLayer(void) { return m_CurrentPlane;} // get the current plane layer
double AutoFocusVolumeScan::getPlaneZpos(int layer) { return m_zPos[layer];} // get the plane position
double AutoFocusVolumeScan::getUpperPlaneZpos(void) {return m_zPos[Upper];} // get the upper plane position
double AutoFocusVolumeScan::getMiddlePlaneZpos(void) { return m_zPos[Middle];}// get the middle plane position
double AutoFocusVolumeScan::getLowerPlaneZpos(void) { return m_zPos[Lower];}// get the lower plane position
double AutoFocusVolumeScan::getFocusValue(int layer){ return m_FocusValue[layer];} // return a focus vaule of the layer

void AutoFocusVolumeScan::setFocusValue(cv::Mat image) {
	vector<double> mean;
	vector<double> std;
	cv::meanStdDev(image, mean, std); // may able to add and ROI

	double std2 = 0;
	for (size_t i = 0; i < std.size(); i++) {
		std2 = std2  + std[i];
	}
	setFocusValue(std2);
}
void AutoFocusVolumeScan::setFocusValue(double FocusValue) {
	m_FocusValue[m_CurrentPlane] = FocusValue;
	m_ImageCompleteFlag[m_CurrentPlane] = true;
	if (m_CurrentPlane == 2) {
		m_VolumeScanCompleteFlag = true;
	}
}

double AutoFocusVolumeScan::getNextScanLayerPosition(void) {
	int i = 0;
	for (i = 0;i<2; i++) {
		if (m_ImageCompleteFlag[m_ScanSequence[i]] == false)
		break;
	}
	m_NextFocalPlane = m_ScanSequence[i];
	return m_zPos[m_NextFocalPlane];
}

double AutoFocusVolumeScan::getNextVolumeScanCenterLayer(void) {
	int maxValueLayer;
	if (m_FocusValue[0] > m_FocusValue[1])
		maxValueLayer = 0;
	else
		maxValueLayer = 1;
	if (m_FocusValue[2] > m_FocusValue[maxValueLayer])
		maxValueLayer = 2;

	return getPlaneZpos(maxValueLayer);
}
// flow chart
// isCompleteVolumeScan();
// Initialization(double zPos);*/

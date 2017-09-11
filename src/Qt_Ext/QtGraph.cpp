#include "Qtgraph.h"
#include "DSP/ImageProcess.h"

Qtgraph::Qtgraph(void)
{
	x = NULL;
	y = NULL;
	gridx = NULL;
	gridy = NULL;
	linecolor = 4;
	gridcolor = 4;
	n = 0;
	idx = 0;
	x0_px = 0; // matched px when x = x0
	y0_px = 0; // matched px when y = x0
	gain_x = 1.0; // matched number of px when x = 1;
	gain_y = 1.0; // matched number of px when y = 1;
	x0 = 0;
	y0 = 0;
	grid_nx = 0;
	grid_ny = 0;
	gridx_max = 0;
	gridy_max = 0;
	enabledraw = false;
	enableupdate = false;
	scale = 1.0;
}

Qtgraph::~Qtgraph()
{
	if (x) free(x);
	if (y) free(y);
	if (gridx) free(gridx);
	if (gridy) free(gridy);
}


void Qtgraph::clearbuffer() {
	if (x) free(x);
	if (y) free(y);
	x = (double *)malloc(n * sizeof(double));
	y = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		y[i] = 0;
	}
}
void Qtgraph::init(int _n, int _x0_px, int _y0_px, double _gain_x, double _gain_y, QColor _linecolor) {
	enabledraw = false;
	enableupdate = false;
	n = _n;
	x0_px = _x0_px;
	y0_px = _y0_px;
	gain_x = _gain_x;
	gain_y = _gain_y;
	linecolor = _linecolor;
	clearbuffer();
}
void Qtgraph::init1D(int _n, int _y0_px, double _gain_x, double _gain_y, QColor _linecolor) {
	n = _n;
	x0_px = 0;
	y0_px = _y0_px;
	gain_x = _gain_x;
	gain_y = _gain_y;
	linecolor = _linecolor;
	clearbuffer();
	for (int i = 0; i < n; i++)
		x[i] = i;
}
void Qtgraph::update(double _x, double _y, int i) {
	if (enableupdate) {
		x[i% n] = _x;
		y[i% n] = _y;
		idx = i;
	}
}
void Qtgraph::update1D(double _y, int i) {
	if (enableupdate) {
		x[i% n] = i% n;
		y[i% n] = _y;
		idx = i;
	}
}
void Qtgraph::setgrid(int _nx, double * _xgrid, int _ny, double * _ygrid, QColor _gridcolor) {
	grid_nx = 0; grid_ny = 0;
	double * gridx_temp;
	double * gridy_temp;
	if (_nx > 0) {
		gridx_temp = (double *)malloc(_nx * sizeof(double));
		gridx_max = _xgrid[0];
		for (int i = 0; i < _nx; i++) {
			gridx_temp[i] = _xgrid[i];
			if (gridx_temp[i] > gridx_max)
				gridx_max = gridx_temp[i];
		}
	}
	if (_ny > 0) {
		gridy_temp = (double *)malloc(_ny * sizeof(double));
		gridy_max = _ygrid[0];
		for (int i = 0; i < _ny; i++) {
			gridy_temp[i] = _ygrid[i];
			if (gridy_temp[i] > gridy_max)
				gridy_max = gridy_temp[i];
		}
	}

	double * gridx_old = gridx;
	double * gridy_old = gridy;
	gridx = gridx_temp;
	gridy = gridy_temp;
	if (gridx_old) { free(gridx_old);}
	if (gridy_old) { free(gridy_old);}

	gridcolor = _gridcolor;
	grid_nx = _nx;
	grid_ny = _ny;
}

void Qtgraph::drawnow(double _x0, double _y0, shared_ptr<QImage> pImage) {
	if (enabledraw) {
		for (int i = 0; i < grid_nx; i++)
			paint_line(Point2d(x0_px + gridx[i] * gain_x, y0_px - gridy[0] * gain_y)*scale, Point2d(x0_px + gridx[i] * gain_x, y0_px - gridy[grid_ny - 1] * gain_y)*scale, gridcolor, pImage);
		for (int i = 0; i < grid_ny; i++)
			paint_line(Point2d(x0_px + gridx[0] * gain_x, y0_px - gridy[i] * gain_y)*scale, Point2d(x0_px + gridx[grid_nx - 1] * gain_x, y0_px - gridy[i] * gain_y)*scale, gridcolor, pImage);

		paint_graph(x, y, n, idx, x0_px, y0_px, gain_x*scale, gain_y*scale, _x0*scale, _y0*scale, linecolor, pImage);
	}
}

void Qtgraph::drawnow(double _x0_px, double _y0_px, double _x0, double _y0, shared_ptr<QImage> pImage) {
	if (enabledraw) {
		for (int i = 0; i < grid_nx; i++)
			paint_line(Point2d(_x0_px + gridx[i] * gain_x, _y0_px - gridy[0] * gain_y)*scale, Point2d(_x0_px + gridx[i] * gain_x, _y0_px - gridy[grid_ny - 1] * gain_y)*scale, gridcolor, pImage);
		for (int i = 0; i < grid_ny; i++)
			paint_line(Point2d(_x0_px + gridx[0] * gain_x, _y0_px - gridy[i] * gain_y)*scale, Point2d(_x0_px + gridx[grid_nx - 1] * gain_x, _y0_px - gridy[i] * gain_y)*scale, gridcolor, pImage);
		paint_graph(x, y, n, idx, _x0_px*scale, _y0_px*scale, gain_x*scale, gain_y*scale, _x0*scale, _y0*scale, linecolor, pImage);

	}
}

void Qtgraph::drawnow1D(double _x0_px, double _y0_px, shared_ptr<QImage> pImage, int offset) {
	if (enabledraw) {
		for (int i = 0; i < grid_nx; i++)
			paint_line(Point2d(_x0_px + gridx[i] * gain_x, _y0_px)*scale, Point2d(_x0_px + gridx[i] * gain_x, _y0_px - gridy_max*gain_y)*scale, gridcolor, pImage);
		for (int i = 0; i < grid_ny; i++)
			paint_line(Point2d(_x0_px, _y0_px - gridy[i] * gain_y)*scale, Point2d(_x0_px + gridx_max*gain_x, _y0_px - gridy[i] * gain_y)*scale, gridcolor, pImage);

		paint_graph(y, n, -1 - offset, _x0_px*scale, _y0_px*scale, (int)gain_x*scale, gain_y*scale, 0.0, linecolor, pImage);
	}
}

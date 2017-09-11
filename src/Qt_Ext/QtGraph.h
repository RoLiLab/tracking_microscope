#ifndef Qtgrqph_H
#define Qtgrqph_H
#include "Base/base.h"
#include "Qt_Ext.h"
#include "Pixmap.h"

class Qtgraph {
public:
	Qtgraph();
	~Qtgraph();
	void init(int _n, int _x0_px, int _y0_px, double _gain_x, double _gain_y, QColor _linecolor);
	void drawnow(double _x0, double _y0, shared_ptr<QImage> pImage);
	void drawnow(double _x0_px, double _y0_px, double _x0, double _y0, shared_ptr<QImage> pImage);
	void drawnow1D(double _x0_px, double _y0_px, shared_ptr<QImage> pImage, int offset);
	void update(double _x, double _y, int i);
	void setgrid(int _nx, double * _xgrid, int _ny, double * _ygrid, QColor _gridcolor);
	void init1D(int _n, int _y0_px, double _gain_x, double _gain_y, QColor _linecolor);
	void update1D(double _y, int i);
	bool enabledraw;
	bool enableupdate;
	double scale;
private:
	double *x;
	double *y;
	int n;
	int idx;
	double x0_px; // matched px when x = x0
	double y0_px; // matched px when y = x0
	double gain_x; // matched number of px when x = 1;
	double gain_y; // matched number of px when y = 1;
	double x0;
	double y0;
	QColor linecolor;
	int grid_nx;
	int grid_ny;
	double * gridx;
	double * gridy;
	double gridx_max;
	double gridy_max;
	QColor gridcolor;
	void clearbuffer();
};

/*enum GlobalColor {
	color0,
	color1,
	black,
	white,
	darkGray,
	gray,
	lightGray,
	red,
	green,
	blue,
	cyan,
	magenta,
	yellow,
	darkRed,
	darkGreen,
	darkBlue,
	darkCyan,
	darkMagenta,
	darkYellow,
	transparent
};*/
#endif // SERIALINTERFACEREADER_H

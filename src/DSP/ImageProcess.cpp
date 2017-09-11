#include "Base/base.h"
#include "DSP/ImageProcess.h"
//#include "device_launch_parameters.h"
#include <numeric>
#include <algorithm>
#include <math.h>

#include "DSP/ImageProcess_NPP.h"
#include "CUDA_Kernels.h"


//------------------------------------------------------------------------------
// function for drawing on QT Images
//------------------------------------------------------------------------------
void paint_cvpoint(Point2i a, const QColor &color, shared_ptr<QImage> pImage) {
	int r = 10;
	QPainter painter( pImage.get() );
	painter.setPen(color);
	painter.drawEllipse(a.x-r/2, a.y-r/2, r, r);
	painter.drawPoint(a.x, a.y);
}
void paint_cvpoint2(Point2i a, const QColor &color, shared_ptr<QImage> pImage) {
	int r = 10;
	QPainter painter( pImage.get() );
	painter.setPen(color);
	painter.drawPoint(a.x, a.y);
}
void paint_cvpoint2(Point2d a, const QColor &color, shared_ptr<QImage> pImage) {
	int r = 10;
	QPainter painter(pImage.get());
	painter.setPen(color);
	painter.drawPoint((int)a.x, (int)a.y);
}
void paint_cvline(Point2i a, double angle_rad, const QColor &color, shared_ptr<QImage> pImage) {
	int r = 10;
	QPainter painter( pImage.get() );
	painter.setPen(color);
	int x = (int)(cos(angle_rad)*r);
	int y = (int)(sin(angle_rad)*r);
	painter.drawLine(a.x, a.y, a.x + x, a.y + y);
}
void paint_cvline(Point2d a, double angle_rad, const QColor &color, shared_ptr<QImage> pImage) {
	int r = 10;
	QPainter painter(pImage.get());
	painter.setPen(color);
	int x = (int)(cos(angle_rad)*r);
	int y = (int)(sin(angle_rad)*r);
	painter.drawLine((int)a.x, (int)a.y, (int)a.x + x, (int)a.y + y);
}

void paint_line(Point2d a, Point2d b, const QColor &color, shared_ptr<QImage> pImage) {
	int r = 10;
	QPainter painter(pImage.get());
	painter.setPen(color);
	painter.drawLine((int)a.x, (int)a.y, (int)b.x, (int)b.y);
}

void paint_line(double * x, double * y, const QColor &color, QImage * pImage) {
	QPainter painter;
	painter.begin(pImage);
	painter.setPen(color);
	painter.drawLine((int)x[0], (int)y[0], (int)x[1], (int)y[1]);
	painter.end();
}

void paint_cvpoint_Rect(Point2i a, const QColor &color, shared_ptr<QImage> pImage) {
	int r = 10;
	QPainter painter( pImage.get() );
	painter.setPen(color);
	painter.drawRect(a.x-r/2, a.y-r/2, r, r);
	painter.drawPoint(a.x, a.y);
}

//void paint_cvROI(cv::Rect roi, const QColor &color, shared_ptr<QImage> pImage) {
//	QPainter painter( pImage.get() );
//	painter.setPen(color);
//	painter.drawRect(roi.x, roi.y, roi.width, roi.height);
//}

void paint_Ellipse(Point2i a, int r, const QColor &color, shared_ptr<QImage> pImage) {
	QPainter painter( pImage.get() );
	painter.setPen(color);
	painter.drawEllipse(a.x-r, a.y-r, 2*r, 2*r);
}
void paint_Ellipse(Point2d a, int r, const QColor &color, shared_ptr<QImage> pImage) {
	QPainter painter(pImage.get());
	painter.setPen(color);
	painter.drawEllipse((int)a.x - r, (int)a.y - r, 2 * r, 2 * r);
}
//
//void paint_Ellipse(cv::RotatedRect a, const QColor &color, shared_ptr<QImage> pImage) {
//	QPainter painter( pImage.get() );
//	painter.setPen(color);
//	painter.translate(a.center.x, a.center.y);
//	painter.rotate(a.angle);
//	painter.drawEllipse(QPointF(0,0), (qreal)a.size.width/2, (qreal)a.size.height/2);
//	painter.restore();
//}

//void paint_RotRect(cv::RotatedRect a, const QColor &color, shared_ptr<QImage> pImage) {
//	QPainter painter( pImage.get() );
//	painter.setPen(color);
//	painter.translate(a.center.x, a.center.y);
//	painter.rotate(a.angle);
//	painter.drawRect(-a.size.width/2, -a.size.height/2, a.size.width, a.size.height);
//	painter.restore();
//}

//void paint_Ellipse(Point2d center, double angle, imSize _size, const QColor &color, shared_ptr<QImage> pImage) {
//	QPainter painter( pImage.get() );
//	painter.setPen(color);
//	painter.translate(center.x, center.y);
//	painter.rotate(angle);
//	painter.drawEllipse(QPointF(0,0), (qreal)_size.width/2, (qreal)_size.height/2);
//	painter.restore();
//}
void paint_graph(double * x, double * y, int n, int n0, int x0, int y0, double xgain, double ygain, double xoffset, double yoffset, const QColor &color, shared_ptr<QImage> pImage) {
	if (n == 0)
		return;
	QPainter painter(pImage.get());
	painter.setPen(color);
	QPolygon polygon = QPolygon(n);
	for (int i = 0; i < n; i++)
		polygon[i] = QPoint(x0 - (int)((x[(n0 - i) % n] - xoffset) * xgain), y0 - (int)((y[(n0 - i) % n] - yoffset) * ygain));
	painter.drawPolyline(polygon);
	//painter.drawPoint(polygon[0]);
	painter.drawEllipse(polygon[0], 3, 3);
}

void paint_graph(double * y, int n, int y0_n, int x0, int y0, int dx, double ygain, double yoffset, const QColor &color, shared_ptr<QImage> pImage) {
	if (n == 0)
		return;
	QPainter painter(pImage.get());
	painter.setPen(color);
	QPolygon polygon = QPolygon(n);
	if (y0_n < 0) {
		for (int i = 0; i < n; i++)
			polygon[i] = QPoint((x0 + dx*(i - 1)), y0 - (int)((y[i] - yoffset) * ygain));
	}
	else {
		for (int i = 0; i < n; i++)
			polygon[i] = QPoint(x0 + dx*(i - 1), y0 - (int)((y[(y0_n - i) % n] - yoffset) * ygain));
	}
	painter.drawPolyline(polygon);
}

void paint_graph(vector<double> y, int x0, int y0, int dx, double ygain, double yoffset, const QColor &color, shared_ptr<QImage> pImage) {
	size_t n = y.size();
	if (n == 0)
		return;
	QPainter painter(pImage.get());
	painter.setPen(color);
	QPolygon polygon = QPolygon(n);
	for (int i = 0; i < n; i++)
		polygon[i] = QPoint(x0 + dx*(i - 1), y0 - (int)((y[i] - yoffset) * ygain));
	painter.drawPolyline(polygon);
}

void paint_Contour(vector<Point2i> contourPoints, const QColor &color, shared_ptr<QImage> pImage, bool closed) {
	size_t n = contourPoints.size();
	if (n == 0)
		return;
	QPainter painter( pImage.get() );
	painter.setPen(color);
	QPolygon polygon = QPolygon(n);
	for (int i = 0; i < n; i++)
		polygon[i] = QPoint(contourPoints[i].x, contourPoints[i].y);

	if (closed)
		painter.drawPolygon(polygon);
	else
		painter.drawPolyline(polygon);
}
void paint_Contour(vector<Point2f> contourPoints, const QColor &color, shared_ptr<QImage> pImage, bool closed) {
	size_t n = contourPoints.size();
	if (n == 0)
		return;
	QPainter painter( pImage.get() );
	painter.setPen(color);
	QPolygon polygon = QPolygon(n);
	for (int i = 0; i < n; i++)
		polygon[i] = QPoint((int)contourPoints[i].x, (int)contourPoints[i].y);

	if (closed)
		painter.drawPolygon(polygon);
	else
		painter.drawPolyline(polygon);
}

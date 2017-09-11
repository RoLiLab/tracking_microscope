#pragma once

// add opencv library - Is there another way to do?

#include "Qt_Ext/Qt_Ext.h"
#include "ipp.h"

//------------------------------------------------------------------------------
// function for drawing on QT Images
//------------------------------------------------------------------------------
void paint_cvpoint(Point2i a, const QColor &color, shared_ptr<QImage> pImage);
void paint_cvpoint2(Point2i a, const QColor &color, shared_ptr<QImage> pImage);
void paint_cvpoint2(Point2d a, const QColor &color, shared_ptr<QImage> pImage);
void paint_cvline(Point2i a, double angle_deg, const QColor &color, shared_ptr<QImage> pImage);
void paint_cvline(Point2d a, double angle_deg, const QColor &color, shared_ptr<QImage> pImage);
void paint_cvpoint_Rect(Point2i a, const QColor &color, shared_ptr<QImage> pImage);
void paint_Ellipse(Point2i a, int r, const QColor &color, shared_ptr<QImage> pImage);
void paint_Ellipse(Point2d a, int r, const QColor &color, shared_ptr<QImage> pImage);
//void paint_Ellipse(cv::RotatedRect a, const QColor &color, shared_ptr<QImage> pImage) ;
//void paint_Ellipse(Point2d center, double angle, imSize _size, const QColor &color, shared_ptr<QImage> pImage);
//void paint_RotRect(cv::RotatedRect a, const QColor &color, shared_ptr<QImage> pImage) ;
void paint_Contour(vector<Point2i> contourPoints, const QColor &color, shared_ptr<QImage> pImage, bool closed);
void paint_Contour(vector<Point2f> contourPoints, const QColor &color, shared_ptr<QImage> pImage, bool closed);
void paint_graph(vector<double> y, int x0, int y0, int dx, double ygain, double yoffset, const QColor &color, shared_ptr<QImage> pImage);
void paint_graph(double * y, int n, int y0_n, int x0, int y0, int dx, double ygain, double yoffset, const QColor &color, shared_ptr<QImage> pImage);
void paint_graph(double * x, double * y, int n, int n0, int x0, int y0, double xgain, double ygain, double xoffset, double yoffset, const QColor &color, shared_ptr<QImage> pImage);
void paint_line(Point2d a, Point2d b, const QColor &color, shared_ptr<QImage> pImage);
void paint_line(double * x, double * y, const QColor &color, QImage * pImage);
//void paint_cvROI(cv::Rect roi, const QColor &color, shared_ptr<QImage> pImage);

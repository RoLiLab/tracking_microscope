///////////////////////////////////////////////////////////
//
// QGLImageViewer.cpp
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "Base/base.h"
#include "QGLImageViewer.h"
#include <QtGui/qimage.h>
#include <QtGui/qpainter.h>


UI::QGLImageViewer::QGLImageViewer( QWidget* pParent/*= nullptr */ ) :
QOpenGLWidget(pParent) {

} // end UI::QGLImageViewer::QGLImageViewer()


QSize UI::QGLImageViewer::GetImageSize() const {
	return m_pImage ? m_pImage->size() : QSize();
} // end UI::QGLImageViewer::GetImageSize()



QRect UI::QGLImageViewer::GetDrawRect() const {
	// Return an empty rect if we have no image
	if( !m_pImage ) return QRect();

	// Grab our zoom rect, use the entire image if empty
	QRect rectZoom= m_rectZoom.isEmpty() ? m_pImage->rect() : m_rectZoom;

	// Preserve our aspect ratio when scaling the image, center the resulting image in our rect
	QSize sizeTarget= rectZoom.size().scaled( size(), Qt::KeepAspectRatio );
	QRect rectTarget( QPoint(0,0), sizeTarget );
	rectTarget.moveCenter( rect().center() );
	return rectTarget;
} // end UI::QGLImageViewer::GetDrawRect()



void UI::QGLImageViewer::SetImage( shared_ptr<QImage> pImage ) {
	// Grab the new image and update us
	m_pImage= pImage;
	update();
} // end UI::QGLImageViewer::SetImage()


void UI::QGLImageViewer::SetZoomRect( const QRect& rectZoom ) {
	// Just return if our zoom rect isn't changing
	if( m_rectZoom == rectZoom ) return;

	// Grab our new zoom rect and redraw
	m_rectZoom= rectZoom;
	update();
} // end UI::QGLImageViewer::SetZoomRect()



void UI::QGLImageViewer::paintEvent( QPaintEvent* ) {
	// Create a painter even if we're not drawing anything else, this erases our background
	QPainter p( this );

	// Just return if we have no image to draw
	if( !m_pImage ) return;

	// Grab our zoom rect, use the entire image if empty, also grab the rect where we're drawing
	QRect rectZoom= m_rectZoom.isEmpty() ? m_pImage->rect() : m_rectZoom,
		  rectTarget= GetDrawRect();

	// Draw our image with smooth scaling
	p.setRenderHint( QPainter::SmoothPixmapTransform, 1 );
	p.drawImage( rectTarget, *m_pImage, rectZoom );
} // end UI::QGLImageViewer::paintEvent()


UI::QGLImageViewer::~QGLImageViewer() {}

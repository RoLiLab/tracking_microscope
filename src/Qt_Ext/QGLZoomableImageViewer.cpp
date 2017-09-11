///////////////////////////////////////////////////////////
//
// QGLZoomableImageViewer.cpp
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "QGLZoomableImageViewer.h"
#include "ZoomRect.h"



namespace {
	const float WHEEL_ZOOM_SENSITIVITY= 1.0f/1000.0f;	//!< The amount we scale our visible rect for each detent of a wheel mouse
	const int MIN_ZOOM_RECT_SIZE= 16;					//!< Our zoom rect width/height can't get smaller than this
} // end file-scope


UI::QGLZoomableImageViewer::QGLZoomableImageViewer( QWidget* pParent/*= nullptr */ ) :
QGLImageViewer(pParent) {

} // end UI::QGLZoomableImageViewer::QGLZoomableImageViewer()


void UI::QGLZoomableImageViewer::mousePressEvent( QMouseEvent* pEvent ) {
	// Ignore the event if we have no image
	if( !HasImage() ) { pEvent->ignore(); return; }

	// On right click, set our zoom in full size
	if( pEvent->buttons() & Qt::RightButton ) {
		if( GetZoomRect().isEmpty() || GetZoomRect().size() == GetImageSize() ) {
			// If we're zoomed out full, reset our zoom in full size
			m_sizeZoomInFull= checked_QSize();
		} else {
			m_sizeZoomInFull= GetZoomRect().size();
		} // end else not zoomed out full
		pEvent->accept();
		return;
	} // end if right click

	// Start a drag at the current cursor location
	m_posDrag= pEvent->pos();
	m_rectDragOriginal= GetZoomRect().isEmpty() ? QRect(QPoint(0,0), GetImageSize()) : GetZoomRect();
	pEvent->accept();

	// added by DH
} // end UI::QGLZoomableImageViewer::mousePressEvent()


void UI::QGLZoomableImageViewer::mouseDoubleClickEvent( QMouseEvent* pEvent ) {
	// Ignore the event if we have no image
	if( !HasImage() ) { pEvent->ignore(); return; }

	// Set a default zoom in full size if necessary
	if( !m_sizeZoomInFull.valid() )
		m_sizeZoomInFull= GetImageSize() / 5;

	// Check whether we're zoomed in at or past our zoom in full size
	if( GetZoomRect().isValid() && GetZoomRect().width() <= m_sizeZoomInFull.cast().width() ) {
		// Zoom out full
		SetZoomRect( QRect(QPoint(0,0), GetImageSize()) );
	} else {
		// Jump to our zoom in size
		float fScale= (float)(m_sizeZoomInFull.cast().width()) / GetImageSize().width();
		zoom( pEvent->pos(), fScale );
	} // end else zooming out full
} // end UI::QGLZoomableImageViewer::mouseDoubleClickEvent()


void UI::QGLZoomableImageViewer::mouseReleaseEvent( QMouseEvent* pEvent ) {
	// Ignore the event if we have no image, or if a drag isn't in progress
	if( !HasImage() || !m_posDrag.valid() ) { pEvent->ignore(); return; }

	// End our drag and accept the event
	m_posDrag= checked_QPoint();
	m_rectDragOriginal= checked_QRect();
	pEvent->accept();
} // end UI::QGLZoomableImageViewer::mouseReleaseEvent()


void UI::QGLZoomableImageViewer::mouseMoveEvent( QMouseEvent* pEvent ) {
	// Ignore the event if we have no image, or if a drag isn't in progress
	if( !HasImage() || !m_posDrag.valid() ) { pEvent->ignore(); return; }

	// Compute our x and y delta in cursor units, then scale them to zoom rect units
	QRect rectZoom= m_rectDragOriginal;
	QPoint posCursorDelta= pEvent->pos() - m_posDrag;
	float fScaleX= (float)(rectZoom.width()) / GetDrawRect().width(),
		  fScaleY= (float)(rectZoom.height()) / GetDrawRect().height();
	QPoint posScaledDelta( (int)(roundf(fScaleX * posCursorDelta.x())), (int)(roundf(fScaleY * posCursorDelta.y())) );

	// Now pan our zoom rect and trim to valid values
	int xAdjusted= rectZoom.x() - posScaledDelta.x(),
		yAdjusted= rectZoom.y() - posScaledDelta.y();
	if( xAdjusted < 0 ) xAdjusted= 0;
	else if( xAdjusted > GetImageSize().width()-rectZoom.width() ) xAdjusted= GetImageSize().width()-rectZoom.width();
	if( yAdjusted < 0 ) yAdjusted= 0;
	else if( yAdjusted > GetImageSize().height()-rectZoom.height() ) yAdjusted= GetImageSize().height()-rectZoom.height();

	// Set our new zoom rect and accept the event
	QRect newWindow = QRect(QPoint(xAdjusted, yAdjusted), rectZoom.size());
	curpos = newWindow.center();
	SetZoomRect(newWindow);
	pEvent->accept();
} // end UI::QGLZoomableImageViewer::mouseMoveEvent()


void UI::QGLZoomableImageViewer::wheelEvent( QWheelEvent* pEvent ) {
	// Ignore the event if we have no image
	if( !HasImage() ) { pEvent->ignore(); return; }

	// Determine our scaling factor and zoom
	float fScale= 1.0f / (1.0f + WHEEL_ZOOM_SENSITIVITY * abs(pEvent->angleDelta().y()));
	if( pEvent->angleDelta().y() < 0.0f ) fScale= 1.0f / fScale;

	zoom(curpos, fScale);

	// Accept this event since we handled it
	pEvent->accept();
} // end UI::QGLZoomableImageViewer::wheelEvent()


void UI::QGLZoomableImageViewer::zoom( const QPoint& posCursor_, float fScale ) {

	// Compute our new zoom rect and set it
	QRect rectCurrent= GetZoomRect().isEmpty() ? QRect(QPoint(0,0), GetImageSize()) : GetZoomRect();
	QPoint posCursor= posCursor_ - GetDrawRect().topLeft();
	QRect rectScaled= Image::ScaledZoomRect( rectCurrent, GetImageSize(), GetDrawRect().size(), posCursor, fScale, MIN_ZOOM_RECT_SIZE );
	curpos = rectScaled.center();
	SetZoomRect( rectScaled );

} // end UI::QGLZoomableImageViewer::zoom()

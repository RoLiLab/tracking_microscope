///////////////////////////////////////////////////////////
//
// QGLImageViewer.h
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Qt_Ext/Qt_Ext.h"
class QImage;


namespace UI {

	//! Draws a single QImage using OpenGL
	class QGLImageViewer : public QOpenGLWidget {
	public:
		//! Initializes us with the given parent widget
		QGLImageViewer( QWidget* pParent= nullptr );

		//! Returns whether we have a valid image
		bool HasImage() const { return m_pImage ? true : false; }
		//! Returns our image size, returns an invalid size if we have no image
		QSize GetImageSize() const;
		//! Returns the rect we use to draw our image, this will preserve our image's aspect ratio and be centered on our canvas
		QRect GetDrawRect() const;

		//! Returns the image we draw, null shared_ptr if none
		shared_ptr<QImage> GetImage() const { return m_pImage; }
		//! Sets the image we draw and updates us
		void SetImage( shared_ptr<QImage> pImage );


		//! Returns our zoom rect, empty if zoomed out all the way
		QRect GetZoomRect() const { return m_rectZoom; }
		//! Sets our zoom rect
		void SetZoomRect( const QRect& rectZoom );

		virtual ~QGLImageViewer();		//!< Non-inline destructor

		bool transposed;
	private:
		//! Draws our image
		virtual void paintEvent( QPaintEvent* pEvent ) override;

		shared_ptr<QImage> m_pImage;	//!< The image we draw on paint events
		QRect m_rectZoom;				//!< The zoom rect which we draw, if empty we draw the entire image
	}; // end class QGLImageViewer

} // end namespace UI

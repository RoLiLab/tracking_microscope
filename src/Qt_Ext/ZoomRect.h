///////////////////////////////////////////////////////////
//
// ZoomRect.h
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#pragma once

class QRect; class QSize; class QPoint;

namespace Image {

	//! Computes a scaled zoom rect given the current zoom rect and:
	//! \a sizeImage - size of original image in pixels
	//! \a sizeRender - size of rendered image on the screen after resampling
	//! \a posCursor - position of cursor within image, with respect to upper-left corner of the rect where the image is rendered
	//! \a fScale - positive scaling factor, 1.0 for no change, > 1 to zoom out (grow zoom rect), < 1 to zoom in (shrink zoom rect)
	//! \a nMinZoomSize - minimum width and height we allow for the zoom rect
	QRect ScaledZoomRect( const QRect& rectCurrent_, const QSize& sizeImage, const QSize& sizeRender, const QPoint& posCursor, float fScale, int nMinZoomSize );

} // end namespace Image

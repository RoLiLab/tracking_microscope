///////////////////////////////////////////////////////////
//
// ZoomRect.cpp
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "Base/base.h"
#include "Qt_Ext/Qt_Ext.h"
#include "Qt_Ext/ZoomRect.h"



QRect Image::ScaledZoomRect( const QRect& rectCurrent_, const QSize& sizeImage, const QSize& sizeRender, const QPoint& posCursor, float fScale, int nMinZoomSize ) {

	// Figure out the size of our new rect, trim to valid values
	QRect rectCurrent= rectCurrent_.isEmpty() ? QRect(QPoint(0,0), sizeImage) : rectCurrent_;
	QSize sizeNew= rectCurrent.size() * fScale;
	if( sizeNew.width() > sizeImage.width() || sizeNew.height() > sizeImage.height() )
		sizeNew= sizeImage;
	if( min(sizeNew.width(), sizeNew.height()) < nMinZoomSize )
		sizeNew= rectCurrent.size();

	// Find the pixel that was under the cursor originally
	int xImage= rectCurrent.left() + (int)(roundf((float)(posCursor.x()) / sizeRender.height() * rectCurrent.height())),
		yImage= rectCurrent.top() + (int)(roundf((float)(posCursor.y()) / sizeRender.height() * rectCurrent.height() ));


	// Find the pixel under the cursor now, adjust our rect to keep the same pixel under the cursor
	int xNew= (int)(roundf( (float)(posCursor.x()) / sizeRender.width() * sizeNew.width() )),
		yNew= (int)(roundf((float)(posCursor.y()) / sizeRender.height() * sizeNew.height())),
		xAdjusted = xImage - xNew,
		yAdjusted = yImage - yNew;

	// Trim to valid values, then return our new zoom rect
	if( xAdjusted < 0 ) xAdjusted= 0;
	else if( xAdjusted > sizeImage.width()-sizeNew.width() ) xAdjusted= sizeImage.width()-sizeNew.width();
	if( yAdjusted < 0 ) yAdjusted= 0;
	else if( yAdjusted > sizeImage.height()-sizeNew.height() ) yAdjusted= sizeImage.height()-sizeNew.height();
	return QRect(QPoint(xAdjusted, yAdjusted), sizeNew);
} // end Image::ScaledZoomRect()

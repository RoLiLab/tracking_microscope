///////////////////////////////////////////////////////////
//
// ZoomRect_tests.cpp: Unit tests for zoom rect manipulation
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "Base/base.h"
#include "Qt_Ext.h"
#include "ZoomRect.h"
#include <gtest/gtest.h>
using namespace Image;


TEST(Qt,ZoomRect) {

	// Zoom in, then out again, about the center of a 640x480 image
	QRect rectScaled= ScaledZoomRect( QRect(0, 0, 640, 480), QSize(640, 480), QSize(640,480), QPoint(320,240), 0.5f, 2 );
	EXPECT_EQ( QRect(160,120,320,240), rectScaled );
	QRect rectOrig= ScaledZoomRect( rectScaled, QSize(640, 480), QSize(640,480), QPoint(320,240), 2.0f, 2 );
	EXPECT_EQ( QRect(0,0,640,480), rectOrig );
} // end zoom rect tests

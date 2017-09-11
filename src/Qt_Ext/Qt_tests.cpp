///////////////////////////////////////////////////////////
//
// Qt_tests.cpp: Qt-related unit tests
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "Base/base.h"
#include "Qt_Ext/Pixmap.h"
#include "Qt_Ext/Qt_Ext.h"
#include <gtest/gtest.h>
#include <random>
using namespace Image;


TEST(Qt,BasicTests) {

	// Create a test image with random data
	const int WIDTH= 77, HEIGHT= 84, STRIDE= 100;
	vector<byte> storage( STRIDE*HEIGHT );
	std::mt19937 rng;
	std::generate( storage.begin(), storage.end(), [&rng]{ return rng(); } );
	GreyscalePixmap pix( storage.data(), WIDTH, HEIGHT, STRIDE );

	// Convert to a QImage, make sure the dimensions and all of the pixels match
	auto pImage= toQImage( pix );
	EXPECT_EQ( WIDTH, pImage->width() );
	EXPECT_EQ( HEIGHT, pImage->height() );
	for( int y=0; y<HEIGHT; ++y ) {
		for( int x=0; x<WIDTH; ++x ) {
			EXPECT_EQ( storage[y*STRIDE+x], qRed(pImage->pixel(x,y)) );
			EXPECT_EQ( storage[y*STRIDE+x], qGreen(pImage->pixel(x,y)) );
			EXPECT_EQ( storage[y*STRIDE+x], qBlue(pImage->pixel(x,y)) );
		} // end for x
	} // end for y

	// Convert back to a pixmap and make sure the result matches
	auto pPixResult= toPixmap( *pImage );
	EXPECT_EQ( pix, *pPixResult );

} // end Qt tests

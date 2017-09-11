///////////////////////////////////////////////////////////
//
// Pixmap.cpp
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "Pixmap.h"
#include "Base/base.h"

Image::GreyscalePixmapConst::GreyscalePixmapConst() :
m_pConstPixels(nullptr),
m_xWidth(0), m_yHeight(0),
m_nStrideBytes(0) {

} // end Image::GreyscalePixmapConst::GreyscalePixmapConst()


Image::GreyscalePixmapConst::GreyscalePixmapConst( const byte* pPixels, int xWidth, int yHeight, int nStrideBytes ) :
m_pConstPixels(pPixels),
m_xWidth(xWidth), m_yHeight(yHeight),
m_nStrideBytes(nStrideBytes) {

} // end Image::GreyscalePixmapConst::GreyscalePixmapConst()


shared_ptr<Image::GreyscalePixmap> Image::GreyscalePixmapConst::Copy() const {
	// Allocate a new pixmap and copy into it
	auto pPixmap= make_shared<GreyscalePixmap>( GetWidth(), GetHeight() );
	CopyTo( *pPixmap );
	return pPixmap;
} // end Image::GreyscalePixmapConst::Copy()


void Image::GreyscalePixmapConst::CopyTo( uint8* pPixels, int xWidth, int yHeight, int nStrideBytes ) const {
	// Make sure we're valid and we match the given params
	//KVERIFYRETURN( m_pConstPixels && pPixels && nStrideBytes >= xWidth && m_xWidth == xWidth && m_yHeight == yHeight );

	// Now copy each row to the given buffer
	const byte* pSource= m_pConstPixels;
	byte* pDest= pPixels;
	for( int y=0; y<GetHeight(); ++y ) {
		copy( pSource, pSource+GetWidth(), pDest );
		pSource+= m_nStrideBytes;
		pDest+= nStrideBytes;
	} // end for y
} // end Image::GreyscalePixmapConst::CopyTo()


void Image::GreyscalePixmapConst::CopyTo( GreyscalePixmap& pixDest ) const {
	return CopyTo( pixDest.GetPixels(), pixDest.GetWidth(), pixDest.GetHeight(), pixDest.GetStrideBytes() );
} // end Image::GreyscalePixmapConst::CopyTo()


bool Image::GreyscalePixmapConst::operator==( const GreyscalePixmapConst& pixOther ) const {
	// If we're both null, return true
	if( !GetPixels() && !pixOther.GetPixels() ) return true;

	// Make sure we're both valid and our width/height match, note that our strides don't need to match
	if( !GetPixels() || !pixOther.GetPixels() ) return false;
	if( GetWidth() != pixOther.GetWidth() || GetHeight() != pixOther.GetHeight() ) return false;

	// Now compare each row, return on mismatch
	const byte* pSource1= GetPixels(), * pSource2= pixOther.GetPixels();
	for( int y=0; y<GetHeight(); ++y ) {
		if( std::memcmp(pSource1, pSource2, GetWidth()) != 0 ) return false;
		pSource1+= GetStrideBytes();
		pSource2+= pixOther.GetStrideBytes();
	} // end for y

	// Pixmaps match
	return true;
} // end Image::GreyscalePixmapConst::operator==()


Image::GreyscalePixmap::GreyscalePixmap() :
GreyscalePixmapConst(nullptr, 0, 0, 0),
m_pPixels(nullptr) {

} // end Image::GreyscalePixmap::GreyscalePixmap()


Image::GreyscalePixmap::GreyscalePixmap( int xWidth, int yHeight ) :
GreyscalePixmapConst(nullptr, xWidth, yHeight, xWidth),
m_storage(xWidth*yHeight),
m_pPixels(m_storage.data()) {
	m_pConstPixels= m_pPixels;		// update our base class's pixel pointer now that we've allocated
} // end Image::GreyscalePixmap::GreyscalePixmap()


Image::GreyscalePixmap::GreyscalePixmap( byte* pPixels, int xWidth, int yHeight, int nStrideBytes ) :
GreyscalePixmapConst(pPixels, xWidth, yHeight, nStrideBytes),
m_pPixels(pPixels) {

} // end Image::GreyscalePixmap::GreyscalePixmap()


void Image::GreyscalePixmap::CopyFrom( const uint8* pPixels, int xWidth, int yHeight, int nStrideBytes ) {
	// Make sure we're valid and we match the given params
	//KVERIFYRETURN( m_pPixels && nStrideBytes >= xWidth && m_xWidth == xWidth && m_yHeight == yHeight );

	// Now copy each row from the given buffer
	const byte* pSource= pPixels;
	byte* pDest= m_pPixels;
	for( int y=0; y<GetHeight(); ++y ) {
		copy( pSource, pSource+GetWidth(), pDest );
		pSource+= nStrideBytes;
		pDest+= m_nStrideBytes;
	} // end for y
} // end Image::GreyscalePixmap::CopyFrom()

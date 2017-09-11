///////////////////////////////////////////////////////////
//
// Qt.cpp
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "Qt_Ext.h"
#include "Pixmap.h"
#include "Base/base.h"


void addMainChild( QWidget* pParent, QWidget* pChild ) {
	// Use a box layout to keep the child filling the parent at all times
	QLayout* pLayout= new QBoxLayout( QBoxLayout::LeftToRight );
	pLayout->addWidget( pChild );
	pParent->setLayout( pLayout );
} // end addMainChild()


void criticalMessageBox( QWidget* pParent, const string& strTitle, const string& strText, const string& strDetailedText ) {
	// We want to make sure our arrow cursor is showing while the messagebox runs
	CursorSentry s_cursor( Qt::ArrowCursor );

	// QMessageBox doesn't provide a static function that can set its detailed text, so we write our own here
	QMessageBox msgbox( pParent );
	msgbox.setWindowTitle( qstr(strTitle) );
	msgbox.setText( qstr(strText) );
	if( !strDetailedText.empty() )
		msgbox.setDetailedText( qstr(strDetailedText) );
	msgbox.setIcon( QMessageBox::Critical );
	(void)msgbox.exec();
} // end criticalMessageBox()



shared_ptr<QImage> toQImage( const Image::GreyscalePixmapConst& pixmap ) {
	// Allocate a new QImage
	auto pImage= make_shared<QImage>( pixmap.GetWidth(), pixmap.GetHeight(), QImage::Format_RGB32 );

	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const byte* pSourceRow= pixmap.GetPixels();
	for( int y=0; y<pixmap.GetHeight(); ++y ) {
		const byte* pSource= pSourceRow;
		QRgb* pDest= reinterpret_cast<QRgb*>( pImage->scanLine(y) );
		for( int x=0; x<pixmap.GetWidth(); ++x ) {
			*pDest++= qRgb( *pSource, *pSource, *pSource );
			++pSource;
		} // end for x
		pSourceRow+= pixmap.GetStrideBytes();
	} // end for y

	return pImage;
} // end toQImage()


shared_ptr<Image::GreyscalePixmap> toPixmap( const QImage& img ) {
	// Allocate a new pixmap
	auto pPixmap= make_shared<Image::GreyscalePixmap>( img.width(), img.height() );

	// Copy each pixel into the pixmap, we can't copy one row at a time because we need to grab the red channel from our 32-bit source
	byte* pDestRow= pPixmap->GetPixels();
	for( int y=0; y<img.height(); ++y ) {
		const QRgb* pSource= reinterpret_cast<const QRgb*>( img.scanLine(y) );
		byte* pDest= pDestRow;
		for( int x=0; x<img.width(); ++x ) {
			*pDest++= static_cast<uint8>( qRed(*pSource) );
			++pSource;
		} // end for x
		pDestRow+= pPixmap->GetStrideBytes();
	} // end for y

	return pPixmap;
} // end toPixmap()



shared_ptr<QImage> Mat2QImage(uchar * src, unsigned int cols, unsigned int rows, unsigned int step) {
		// Allocate a new QImage
		auto pImage= make_shared<QImage>( cols, rows, QImage::Format_RGB32 );

		for( int y=0; y<rows; ++y ) {
			uchar * pSource= src;
			QRgb* pDest= reinterpret_cast<QRgb*>( pImage->scanLine(y) );
			for( int x=0; x<cols; ++x ) {
				if (pSource == NULL) return NULL;
				*pDest++= qRgb( *pSource, *pSource, *pSource );
				++pSource;
			} // end for x
			src+= step;
		} // end for y

		return pImage;
} // end toQImage()



shared_ptr<QImage> Mat2QImage3(uchar * src, unsigned int cols, unsigned int rows, unsigned int step) {
		// Allocate a new QImage
		auto pImage= make_shared<QImage>( cols, rows, QImage::Format_RGB32 );

		for( int y=0; y<rows; ++y ) {
			uchar * pSource= src;
			QRgb* pDest= reinterpret_cast<QRgb*>( pImage->scanLine(y) );
			for( int x=0; x<cols; ++x ) {
				if (pSource == NULL) return NULL;
				*pDest++= qRgb( *pSource, *(pSource+1), *(pSource+2) );
				pSource += 3;
			} // end for x
			src+= step;
		} // end for y

		return pImage;
} // end toQImage()

shared_ptr<QImage> toQImage(const img_uint8 * im) {
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const byte * pSourceRow = im->data;
	for (int y = 0; y<im->imgSize.height; ++y) {
		const byte* pSource = pSourceRow;
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; ++x) {
				*pDest++ = qRgb(*pSource, *pSource, *pSource);
			++pSource;
		} // end for x
		pSourceRow += im->imgSize.width;
	} // end for y

	return pImage;
} // end toQImage()

shared_ptr<QImage> toQImage(const img_uint8 * im, QImage * im_inset, bool enable) {
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const byte * pSourceRow = im->data;
	for (int y = 0; y<im->imgSize.height; ++y) {
		const byte* pSource = pSourceRow;
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; ++x) {
			if (enable && x >= 0 && x < im_inset->width() && y >= 0 && y < im_inset->height()) {
				int temp = im_inset->pixel(x, y);
				if (temp > 0)
					temp = 255;
				else
					temp = *pSource;
				*pDest++ = qRgb(*pSource, temp, *pSource);
			}
			else
				*pDest++ = qRgb(*pSource, *pSource, *pSource);
			++pSource;
		} // end for x
		pSourceRow += im->imgSize.width;
	} // end for y

	return pImage;
} // end toQImage()

shared_ptr<QImage> toQImage(const img_uint16 * im) {
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const uint16 * pSourceRow = im->data;
	for (int y = 0; y<im->imgSize.height; ++y) {
		const uint16 * pSource = pSourceRow;
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; ++x) {
			*pDest++ = qRgb(*pSource, *pSource, *pSource);
			++pSource;
		} // end for x
		pSourceRow += sizeof(uint16)*im->imgSize.width;
	} // end for y

	return pImage;
} // end toQImage()

shared_ptr<QImage> toQImage(const img_uint16 * im, uint16 _min, uint16 _max) {
	if (_max <= _min)
		_max = _min + 1;
	double slope = (float)(_max - _min);
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const uint16 * pSourceRow = im->data;

	for (int y = 0; y<im->imgSize.height; ++y) {
		const uint16 * pSource = pSourceRow;
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; ++x) {
			uint16 intensity = 0;
			if (*pSource > _max)
				intensity = UINT8_MAX;
			else if (*pSource < _min)
				intensity = 0;
			else
				intensity = (uint16)(((double)(*pSource - _min)) / slope * UINT8_MAX);
			*pDest++ = qRgb((int)intensity, (int)intensity, (int)intensity);
			++pSource;
		} // end for x
		pSourceRow += im->imgSize.width;
	}

	// end for y

	return pImage;
} // end toQImage()


shared_ptr<QImage> toQImage_shift(const img_uint16 * im, uint16 _min, uint16 _max, int bit, bool colorcode) {
	if (_max <= _min)
		_max = _min + 1;
	double slope = (float)(_max - _min);
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.height, im->imgSize.width, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const uint16 * pSourceRow = im->data;

	for (int x = 0; x<im->imgSize.width; x++) {
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(x));
		for (int y = 0; y<im->imgSize.height; y++) {
			uint16 intensity = (im->data[(im->imgSize.width - 1 - x) + y*im->imgSize.width]) >> bit;
			if (intensity >= 4095) {
				if (colorcode)
					*pDest++ = qRgb(UINT8_MAX, 0, 0);
				else
					*pDest++ = qRgb(UINT8_MAX, UINT8_MAX, UINT8_MAX);
			}
			else if (intensity > _max) {
				if (colorcode)
					*pDest++ = qRgb(0, 0, UINT8_MAX);
				else
					*pDest++ = qRgb(UINT8_MAX, UINT8_MAX, UINT8_MAX);
			}

			else if (intensity < _min){
				if (colorcode)
					*pDest++ = qRgb(0, UINT8_MAX, 0);
				else
					*pDest++ = qRgb(0, 0, 0);
			}
			else {
				intensity = (uint16)(((double)(intensity - _min)) / slope * UINT8_MAX);
				*pDest++ = qRgb((int)intensity, (int)intensity, (int)intensity);
			}
		} // end for x
	}
	return pImage;
} // end toQImage()

shared_ptr<QImage> toQImage_tf(const img_uint16 * im, uint16 _min, uint16 _max) {
	if (_max <= _min)
		_max = _min + 1;
	double slope = (float)(_max - _min);
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.height, im->imgSize.width, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const uint16 * pSourceRow = im->data;

	for (int y = 0; y<im->imgSize.width; ++y) {
		const uint16 * pSource = pSourceRow;
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.height; ++x) {
			if (*pSource == 4095)
				*pDest++ = qRgb(UINT8_MAX, 0, 0);
			else {
				if (*pSource > _max)
					*pDest++ = qRgb(0, UINT8_MAX, 0); // yellow
				else if (*pSource < _min)
					*pDest++ = qRgb(0, 0, UINT8_MAX);
				else {
					uint16 intensity = (uint16)(((double)(*pSource - _min)) / slope * UINT8_MAX);
					*pDest++ = qRgb((int)intensity, (int)intensity, (int)intensity);
				}
			}
			pSource += im->imgSize.width;
		} // end for x
		pSourceRow += 1;
	}
	// end for y

	return pImage;
} // end toQImage()


shared_ptr<QImage> toQImage_u12(const img_uint16 * im, uint16 _min, uint16 _max) {
	if (_max <= _min)
		_max = _min + 1;
	double slope = (float)(_max - _min);
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	unsigned char * pSourceRow = (unsigned char *)im->data;
	unsigned char * pSource = pSourceRow;
	for (int y = 0; y<im->imgSize.height; ++y) {
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; x = x + 2) {

			unsigned char * b0 = pSource;
			unsigned char * b1 = pSource + 1;
			unsigned char * b2 = pSource + 2;
			uint16 v1 = ((uint16)(*b0) << 4) | (*b1 & 0x0F);
			uint16 v2 = ((uint16)(*b2) << 4) | (*b1 & 0xF0);

			unsigned char intensity = 0;
			if (v1 > _max) intensity = UINT8_MAX;
			else if (v1  < _min) intensity = 0;
			else
				intensity = (uint16)(((double)(v1 - _min)) / slope * UINT8_MAX);
			*pDest = qRgb((int)intensity, (int)intensity, (int)intensity);
			pDest += 1;

			intensity = 0;
			if (v2  > _max) intensity = UINT8_MAX;
			else if (v2  < _min) intensity = 0;
			else
				intensity = (uint16)(((double)(v2 - _min)) / slope * UINT8_MAX);
			*pDest = qRgb((int)intensity, (int)intensity, (int)intensity);
			pDest += 1;
			pSource += 3;
		} // end for x
	} // end for y

	return pImage;
} // end toQImage()

shared_ptr<QImage> toQImage_u12(const img_uint8 * im, uint16 _min, uint16 _max) {
	if (_max <= _min)
		_max = _min + 1;
	double slope = (float)(_max - _min);
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	unsigned char * pSourceRow = (unsigned char *)im->data;
	unsigned char * pSource = pSourceRow;
	for (int y = 0; y<im->imgSize.height; ++y) {
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; x = x + 2) {

			unsigned char * b0 = pSource;
			unsigned char * b1 = pSource + 1;
			unsigned char * b2 = pSource + 2;
			uint16 v1 = ((uint16)(*b0) << 4) | (*b1 & 0x0F);
			uint16 v2 = ((uint16)(*b2) << 4) | (*b1 & 0xF0);

			unsigned char intensity = 0;
			if (v1 > _max) intensity = UINT8_MAX;
			else if (v1  < _min) intensity = 0;
			else
				intensity = (uint16)(((double)(v1 - _min)) / slope * UINT8_MAX);
			*pDest = qRgb((int)intensity, (int)intensity, (int)intensity);
			pDest += 1;

			intensity = 0;
			if (v2  > _max) intensity = UINT8_MAX;
			else if (v2  < _min) intensity = 0;
			else
				intensity = (uint16)(((double)(v2 - _min)) / slope * UINT8_MAX);
			*pDest = qRgb((int)intensity, (int)intensity, (int)intensity);
			pDest += 1;
			pSource += 3;
		} // end for x
	} // end for y

	return pImage;
} // end toQImage()


shared_ptr<QImage> toQImage_u10(const img_uint16 * im, uint16 _min, uint16 _max) {
	if (_max <= _min)
		_max = _min + 1;
	double slope = (float)(_max - _min);
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const uint16 * pSourceRow = im->data;

	for (int y = 0; y<im->imgSize.height; ++y) {
		const uint16 * pSource = pSourceRow;
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; ++x) {
			uint16 intensity0 = (*pSource) >> 6;
			uint16 intensity = 0;
			if (intensity0 > _max)
				intensity = UINT8_MAX;
			else if (intensity0 < _min)
				intensity = 0;
			else
				intensity = (uint16)(((double)(intensity0 - _min)) / slope * UINT8_MAX);
			*pDest++ = qRgb((int)intensity, (int)intensity, (int)intensity);
			++pSource;
		} // end for x
		pSourceRow += im->imgSize.width;
	}

	// end for y

	return pImage;
} // end toQImage()

shared_ptr<QImage> toQImage_replay(const img_uint16 * im, uint16 _min, uint16 _max) {
	if (_max <= _min)
		_max = _min + 1;
	double slope = (float)(_max - _min);
	// Allocate a new QImage
	auto pImage = make_shared<QImage>(im->imgSize.width, im->imgSize.height, QImage::Format_RGB32);
	// Copy each pixel into the image, we can't copy one row at a time because we need to synthesize each 32-bit pixel from our
	// 8-bit source. QImage doesn't seem to have an 8-bit greyscale mode
	const uint16 * pSourceRow = im->data;
	for (int y = 0; y<im->imgSize.height; ++y) {
		const uint16 * pSource = pSourceRow;
		QRgb* pDest = reinterpret_cast<QRgb*>(pImage->scanLine(y));
		for (int x = 0; x<im->imgSize.width; ++x) {
			uint16 intensity = 0;
			if (*pSource > _max)
				intensity = UINT16_MAX;
			else if (*pSource < _min)
				intensity = 0;
			else
				intensity = (uint16)(((double)(*pSource - _min)) / slope * UINT16_MAX);
			*pDest++ = qRgb((int)intensity, (int)intensity, (int)intensity);
			++pSource;
		} // end for x
		pSourceRow += im->imgSize.width;
	} // end for y

	return pImage;
} // end toQImage()

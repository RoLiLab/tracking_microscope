///////////////////////////////////////////////////////////
//
// Qt.h: Imports all Qt headers, suppresses warnings as necessary, also some utility functions
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#pragma once


// Qt generates a few warnings so we suppress them here
#ifdef __KPLATFORM_WIN__
	__pragma(warning(push))
	__pragma(warning(disable: 4127))	// conditional expression is constant, Qt seems to allow this
	__pragma(warning(disable: 4251))	// QGLContext::d_ptr class QScopedPointer<T> needs dll-interfce, we just compile QScopedPointer<T> locally and it seems to work (not sure if this can bite us if we change compilers?)
#endif // end if win

#include "Base/base.h"
#include <QtCore/QtCore>
#include <QtGui/QtGui>
#include <QtWidgets/QtWidgets>
#include <QtWidgets/QOpenGLWidget>


#ifdef __KPLATFORM_WIN__
	__pragma(warning(pop))
#endif // end if win

//! Converts a \c QString into a UTF-8 encoded \c std::string
inline string str( const QString& s ) { return string(s.toUtf8().constData()); }

//! Converts a \c std::string into a \c QString
inline QString qstr( const std::string& s ) { return QString(s.c_str()); }
//! Converts a \c path into a \c QString
//inline QString qstr( const path& p ) { return QString(str(p).c_str()); }
////! Converts a \c boost::format into a \c QString
//inline QString qstr( const boost::basic_format<char>& f ) { return QString(f.str().c_str()); }

//! Adds a child to \pParent with a layout making the child fill the entire parent rect at all times
void addMainChild( QWidget* pParent, QWidget* pChild );

//! Shows a critical messagebox with the given title, text and detailed text
void criticalMessageBox( QWidget* pParent, const string& strTitle, const string& strText, const string& strDetailedText= string() );

namespace Image { class GreyscalePixmap; class GreyscalePixmapConst; }
//! Converts a pixmap into a \c QImage
shared_ptr<QImage> toQImage( const Image::GreyscalePixmapConst& pixmap );
shared_ptr<QImage> toQImage(const img_uint8 * im);
shared_ptr<QImage> toQImage(const img_uint8 * im, QImage * im_inset, bool enable);
shared_ptr<QImage> toQImage(const img_uint16 * im);
shared_ptr<QImage> toQImage(const img_uint16 * im, uint16 _min, uint16 _max);
shared_ptr<QImage> toQImage_shift(const img_uint16 * im, uint16 _min, uint16 _max, int bit, bool colorcode);
shared_ptr<QImage> toQImage_u12(const img_uint8 * im, uint16 _min, uint16 _max);
shared_ptr<QImage> toQImage_u12(const img_uint16 * im, uint16 _min, uint16 _max);
shared_ptr<QImage> toQImage_u10(const img_uint16 * im, uint16 _min, uint16 _max);
shared_ptr<QImage> toQImage_replay(const img_uint16 * im, uint16 _min, uint16 _max);
shared_ptr<QImage> toQImage_tf(const img_uint16 * im, uint16 _min, uint16 _max);

//! Converts a \c QImage into a pixmap
shared_ptr<Image::GreyscalePixmap> toPixmap( const QImage& img );

shared_ptr<QImage> Mat2QImage(uchar * src, unsigned int cols, unsigned int rows, unsigned int step);
shared_ptr<QImage> Mat2QImage3(uchar * src, unsigned int cols, unsigned int rows, unsigned int step);


//! Stack-based sentry which shows a cursor for its duration
class CursorSentry {
public:
	explicit CursorSentry( Qt::CursorShape cursorShape= Qt::WaitCursor ) { QGuiApplication::setOverrideCursor( QCursor(cursorShape) ); }
	~CursorSentry() { QGuiApplication::restoreOverrideCursor(); }
}; // end class CursorSentry

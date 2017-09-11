///////////////////////////////////////////////////////////
//
// Pixmap.h: 8-bit greyscale only for now, we can add a 32-bit RGBA version (and perhaps a common base class) if needed in the future
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#pragma once
#include <memory>
#include <vector>
typedef unsigned char byte;

namespace Image {
	class GreyscalePixmap;


	//! An 8-bit greyscale image represented by a vector of bytes. We can allocate our own storage or use
	//! client-provided storage (in which case we don't deallocate it, we simply store pointers
	//! into it). This base class version is read-only
	class GreyscalePixmapConst {
    public:
		//! Creates an empty read-only greyscale pixmap
		GreyscalePixmapConst();
		//! Creates a greyscale pixmap with the given storage
		GreyscalePixmapConst( const byte * pPixels, int xWidth, int yHeight, int nStrideBytes );

		int GetWidth() const { return m_xWidth; }		//!< Returns our width
		int GetHeight() const { return m_yHeight; }		//!< Returns our height
		int GetStrideBytes() const { return m_nStrideBytes; }	//!< Returns our stride, in bytes

		const byte* GetPixels() const { return m_pConstPixels; }	//!< Returns our pixels, null if we have none

		//! Allocates a new non-const pixmap and copies us to it
		std::shared_ptr<GreyscalePixmap> Copy() const;

		//! Copies our pixels to the given buffer
		void CopyTo( uint8_t* pPixels, int xWidth, int yHeight, int nStrideBytes ) const;
		//! Copies our pixels to the given pixmap
		void CopyTo( GreyscalePixmap& pixDest ) const;

		//! Compares our dimensions and pixels to the given pixmap
		bool operator==( const GreyscalePixmapConst& pixOther ) const;

    protected:
		const byte* m_pConstPixels;		//!< Pointer to our pixels, null if we're empty, otherwise pointing to derived class storage or user-provided storage
		int m_xWidth, m_yHeight,		//!< Our dimensions, defaults to 0x0
			m_nStrideBytes;			//!< Offset in bytes between rows in our storage
	}; // end class GreyscalePixmapConst


	//! An 8-bit greyscale image that can be read or written
	class GreyscalePixmap : public GreyscalePixmapConst {
    public:
		//! Creates an empty greyscale pixmap
		GreyscalePixmap();
		//! Allocates storage for a greyscale pixmap with the given dimensions, our stride equals our width in this case
		GreyscalePixmap( int xWidth, int yHeight );
		//! Creates a greyscale pixmap with the given read/write user-provided storage
		GreyscalePixmap( byte* pPixels, int xWidth, int yHeight, int nStrideBytes );

		//! Creation function, returns a shared_ptr
		static std::shared_ptr<GreyscalePixmap> Create( int xWidth, int yHeight ) { return std::shared_ptr<GreyscalePixmap>( new GreyscalePixmap(xWidth, yHeight) ); }

		byte* GetPixels() { return m_pPixels; }	//!< Returns our pixels, null if we're empty

		//! Copies our pixels from the given buffer
		void CopyFrom( const uint8_t * pPixels, int xWidth, int yHeight, int nStrideBytes );
		//! Copies our pixels from the given pixmap
		void CopyFrom( const GreyscalePixmapConst& pixSource ) { return CopyFrom( pixSource.GreyscalePixmapConst::GetPixels(), pixSource.GetWidth(), pixSource.GetHeight(), pixSource.GetStrideBytes() ); }


    private:
		std::vector<byte> m_storage;		//!< Storage for our pixels, empty if using user-allocated storage
		byte* m_pPixels;			//!< Pointer to our writable pixels, null if we're empty or read-only, can point into \c m_storage or user-provided storage
	}; // end class GreyscalePixmap

} // end namespace Image

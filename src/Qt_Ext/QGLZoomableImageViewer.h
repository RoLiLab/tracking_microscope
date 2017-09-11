///////////////////////////////////////////////////////////
//
// QGLZoomableImageViewer.h
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#pragma once
#include "QGLImageViewer.h"
class QImage;

template<typename T> class checked_value {
public:
	//! When initialized by default we take \c T's default value and an invalid state
	checked_value() : m_val(T()), m_bValid(false) {}

	//! When initialized or implicitly converted we take the given value and a valid state
	checked_value(T val) : m_val(std::move(val)), m_bValid(true) {}

	//! Explicitly specify our value and default state
	checked_value(T val, bool bValid) : m_val(std::move(val)), m_bValid(bValid) {}

	//! Copy-construct from a checked-value
	checked_value(const checked_value<T>& other) :
		m_val(other.m_val),
		m_bValid(other.m_bValid) {}

	//! Move-construct from a temporary checked value
	checked_value(checked_value<T>&& other) :
		m_val(std::move(other.m_val)),
		m_bValid(other.m_bValid) {}

	//! Initialize from a checked value of another type
	template <typename U>
	explicit checked_value(const checked_value<U>& val) :
		m_val(val.IsValid() ? T(val.cast()) : T()),
		m_bValid(val.IsValid()) {}

	//! Returns our value, asserts our validity
	inline operator T() const { /*KASSERT(IsValid());*/ return m_val; }
	//! Set and validate our value
	inline checked_value<T>& operator=(const T& val) { m_val = val; SetValid(true); return *this; }
	//! Set and validate our value, from a temporary
	inline checked_value<T>& operator=(T&& val) { m_val = std::move(val); SetValid(true); return *this; }

	//! Grab \a checked_val's value and validity flag
	inline checked_value<T>& operator=(const checked_value<T>& checked_val) { m_val = checked_val.m_val; SetValid(checked_val.m_bValid); return *this; }	//lint !e1529 "self-assignment is safe"
																																							//! Grab \a checked_val's value and validity flag, from a temporary
	inline checked_value<T>& operator=(checked_value<T>&& checked_val) { m_val = std::move(checked_val.m_val); SetValid(checked_val.m_bValid); return *this; }	//lint !e1529 "self-assignment is safe"

																																								//! Return our value itself so its members can be accessed
	inline T& get() { return m_val; }	//lint !e1536 "intentionally exposing low access member"
										//! Return our value itself so its members can be accessed
	inline const T& get() const { return m_val; }

	//! Return our value itself so its members can be accessed, asserts validity
	inline T& cast() { /*KASSERT(IsValid());*/ return m_val; }	//lint !e1536 "intentionally exposing low access member"
															//! Return our value itself so its members can be accessed, asserts validity
	inline const T& cast() const { /*KASSERT(IsValid());*/ return m_val; }

	//! Return our value, or a provided default value if we're not valid.
	inline T otherwise(T defaultVal) const { return IsValid() ? cast() : defaultVal; }

	//! Returns a pointer to our value if it's valid, otherwise returns NULL
	T* ptr() { return IsValid() ? &m_val : NULL; }
	//! Returns a const pointer to our value if it's valid, otherwise returns NULL
	const T* ptr() const { return IsValid() ? &m_val : NULL; }

	inline checked_value<T>& operator++() { /*KASSERT(IsValid());*/ ++m_val; return *this; }	//!< Prefix increment
																							//! Postfix increment
	inline const checked_value<T> operator++(int) {
		//KASSERT(IsValid());
		checked_value<T> old = *this;
		++(*this);
		return old;
	} // end operator++

	inline checked_value<T>& operator--() { /*KASSERT(IsValid());*/ --m_val; return *this; }	//!< Prefix decrement
																							//! Postfix decrement
	inline const checked_value<T> operator--(int) {
		//KASSERT(IsValid());
		checked_value<T> old = *this;
		--(*this);
		return old;
	} // end operator--

	  //! Self-increment
	inline checked_value<T>& operator+=(const checked_value<T>& val) {
		//KASSERT(IsValid());
		m_val += val.m_val;
		return *this;
	} // end operator+=
	  //! Self-decrement
	inline checked_value<T>& operator-=(const checked_value<T>& val) {
		//KASSERT(IsValid());
		m_val -= val.m_val;
		return *this;
	} // end operator-=

	  //! Self-multiply
	inline checked_value<T>& operator*=(const checked_value<T>& val) {
		//KASSERT(IsValid());
		m_val *= val.m_val;
		return *this;
	} // end operator*=
	  //! Self-divide
	inline checked_value<T>& operator/=(const checked_value<T>& val) {
		//KASSERT(IsValid());
		m_val /= val.m_val;
		return *this;
	} // end operator/=

	  //! Self-modulus
	inline checked_value<T>& operator%=(const checked_value<T>& val) {
		//KASSERT(IsValid());
		m_val %= val.m_val;
		return *this;
	} // end operator%=

	  //! Self-left-shift
	inline checked_value<T>& operator<<=(const checked_value<T>& val) {
		//KASSERT(IsValid());
		m_val <<= val.m_val;
		return *this;
	} // end operator<<=
	  //! Self-right-shift
	inline checked_value<T>& operator>>=(const checked_value<T>& val) {
		////KASSERT(IsValid());
		m_val >>= val.m_val;
		return *this;
	} // end operator>>=

	  //! Self-logical and
	inline checked_value<T>& operator&=(const checked_value<T>& val) {
		////KASSERT(IsValid());
		m_val &= val.m_val;
		return *this;
	} // end operator&=
	  //! Self-logical or
	inline checked_value<T>& operator|=(const checked_value<T>& val) {
		////KASSERT(IsValid());
		m_val |= val.m_val;
		return *this;
	} // end operator|=
	  //! Self-logical invert
	inline checked_value<T>& operator^=(const checked_value<T>& val) {
		////KASSERT(IsValid());
		m_val ^= val.m_val;
		return *this;
	} // end operator^=


	inline bool IsValid() const { return m_bValid; }	//!< Returns TRUE if we're valid, returns FALSE otherwise
	inline bool valid() const { return IsValid(); }		//!< Same as \c IsValid(), more in line with STL naming conventions
	inline void SetValid(bool bValid) { m_bValid = bValid; }	//!< Changes our valid state to \a bValid

	inline void Validate() { SetValid(true); }		//!< Sets our valid state
													//! Sets our valid state and sets our value to \a val, same as \c operator=
	inline void Validate(const T& val) { SetValid(true); m_val = val; }

	inline void Invalidate() { SetValid(false); }	//!< Clears our valid state
													//! Clears our valid state and sets our value to \a val
	inline void Invalidate(const T& val) { SetValid(false); m_val = val; }

private:
	T m_val;		//!< Our actual value
	bool m_bValid;	//!< Set when we are valid
}; // end class checked_value<>


//! A checked \c QPoint
typedef checked_value<QPoint> checked_QPoint;
//! A checked \c QSize
typedef checked_value<QSize> checked_QSize;
//! A checked \c QRect
typedef checked_value<QRect> checked_QRect;


namespace UI {

	//! Inherits \c QGLImageViewer, adds mouse-based zoom/pan functionality
	class QGLZoomableImageViewer : public QGLImageViewer {
	public:
		//! Initializes us with the given parent widget
		QGLZoomableImageViewer( QWidget* pParent= nullptr );
		QPoint curpos;
	private:
		checked_QPoint m_posDrag;			//!< The start of the current drag operation, invalid if no drag in progress
		checked_QRect m_rectDragOriginal;	//!< Our zoom rect at the start of the current drag operation, invalid if no drag in progress
		checked_QSize m_sizeZoomInFull;		//!< Our zoom in full size, we jump to it on double-click if zoomed out full

		virtual void mousePressEvent( QMouseEvent* pEvent );		//!< Start a drag if necessary
		virtual void mouseDoubleClickEvent( QMouseEvent* pEvent );	//!< Zoom in/out full
		virtual void mouseReleaseEvent( QMouseEvent* pEvent );		//!< Zoom in or out if necessary
		virtual void mouseMoveEvent( QMouseEvent* pEvent );			//!< Handle a drag if necessary
		virtual void wheelEvent( QWheelEvent* pEvent );				//!< Zoom in or out

		void zoom( const QPoint& posCursor_, float fScale );			//!< Zooms in/out by scaling factor \a fScale
	}; // end class QGLZoomableImageViewer

} // end namespace UI

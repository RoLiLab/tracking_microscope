///////////////////////////////////////////////////////////
//
// Queue_lockfree.h
// Copyright (c) 2014 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#pragma once

//#pragma message (__FILE__ "(" STRINGIZE(__LINE__) ") : Work in progress") // rewrite from scratch, use explicit memory ordering when writing atomic types, look into which ordering we need here

#include <mutex>
#include <atomic>
#include "boost/noncopyable.hpp"

namespace LockFree {

	//! Single-producer, single-consumer lock-free queue (FIFO), with several enhancements:
	//! - If growing is enabled, when pushing to a full queue, the queue locks and grows (thread-safe and blocking)
	//! - If multiple readers or writers are used, the reads and writes will be serialized, but simultaneous reads and writes are non-blocking
	template<typename T> class Queue : noncopyable {
	public:
		//! Initialize, reserve space for \a uReserve elements, if \a bGrowable is TRUE then we grow, otherwise push can fail when full
		Queue( size_t uReserve, bool bGrowable= true ) :
			m_bGrowable(bGrowable),
			m_iRead(0), m_iWrite(0), m_items(uReserve+1) {}

		//! Returns whether we grow when pushing into a full queue
		bool IsGrowable() const { return m_bGrowable; }
		//! Sets whether we grow when pushing into a full queue
		void SetGrowable( bool bGrowable ) { m_bGrowable= bGrowable; }

		//! Returns the number of items we can hold, note that this method is non-blocking but can fail during a grow
		//! operation, hence the "Unsafe" suffix in the same
		size_t CapacityUnsafe() const { KVERIFYRETURN( !m_items.empty(), 0 ); return m_items.size()-1; }

		//! Returns whether we're empty, note that this is serialized with \c Pop() operations
		bool IsEmpty() const {
			// This is considered a read operation since it's of interest to the reading thread, so hold our read lock
			std::lock_guard<std::mutex> s_lockRead( m_lockRead );

			// We're empty when our read/write indices are equal
			if( m_iRead.load() == m_iWrite.load() ) return true;
			else return false;
		} // end IsEmpty()


		//! Pushes an element into the queue, grows if necessary. Returns TRUE on success, returns FALSE if we're not growable and we're full
		bool Push( const T& item ) {
			// Hold our write lock when pushing
			std::lock_guard<std::mutex> s_lockWrite( m_lockWrite );

			// Attempt to write an element
			while( true ) {
				// Grab our indices
				size_t iRead= m_iRead.load(), iWrite= m_iWrite.load();

				// Compute our next write index
				size_t iWriteNext= iWrite + 1;
				if( iWriteNext >= m_items.size() ) iWriteNext= 0;

				// Check for overflow
				if( iWriteNext == iRead ) {
					// Grow and retry if necessary
					if( m_bGrowable ) {
						growFromPush();
						continue;
					} else {
						// We're not growable and we're full, so fail
						return false;
					} // end else not growable
				} // end if overflow

				// Write our element, then update our index. Note that the order here
				// is critical, and we should probably investigate memory barriers
				// to get this right in the future
				m_items[iWrite]= item;
				m_iWrite.store( iWriteNext );
				return true;
			} // end while attempting write
		} // end Push()

		bool Pop( T& item ) {
			// Popping is a read operation, so hold our read lock
			std::lock_guard<std::mutex> s_lockRead( m_lockRead );

			// Attempt to read an element
			while( true ) {
				// Grab our indices, check for empty
				size_t iRead= m_iRead.load(), iWrite= m_iWrite.load();
				if( iRead == iWrite ) return false;

				// Read the current item
				T itemTemp= m_items[iRead];

				// Attempt to advance the index, use weak compare exchange since we're already in a loop and this can improve
				// performance (http://en.cppreference.com/w/cpp/atomic/atomic_compare_exchange)
				size_t iReadNext= iRead >= m_items.size()-1 ? 0 : iRead+1,
					   iExpected= iRead;
				bool bSuccess= atomic_compare_exchange_weak( &m_iRead, &iExpected, iReadNext );

				// If the CAS operation succeeded then return the value we read, otherwise loop and try again
				if( bSuccess ) {
					item= itemTemp;
					return true;
				} // end if success
			} // end while attempting read
		} // end Pop()



	private:
		bool m_bGrowable;		//!< When set, we block and allocate on push into a full queue. When clear, we fail on push into a full queue

		vector<T> m_items;		//!< Our buffer, allocated in our constructor and when we grow

		atomic<size_t> m_iRead, m_iWrite;		//!< The index where the next read/write will occur, there are m_iWrite-m_iRead items in the queue, so we can't distinguish m_items.size() items from 0 items, so m_items.size()-1 is our capacity

		std::mutex m_lockRead, m_lockWrite;		//!< Locks held on reading/writing threads, both locks are held when we grow


		//! Lock both our read and write locks and grow our FIFO
		void growFromPush() {
			// This is only called by \c Push(), so our write lock is already held. Just lock our read lock here
			std::lock_guard<std::mutex> s_lockRead( m_lockRead );

			vector<T> newItems( m_items.size()*2 ); // a new vector for holding our items, appropriately sized

			// If there are any elements after the read index, they need to be moved since they'd be clobbered
			typename vector<T>::iterator iNewEnd= copy( m_items.begin()+m_iRead.load(), m_items.end(), newItems.begin() );

			// Now we append the elements up to, but not including, the write head
			copy( m_items.begin(), m_items.begin()+m_iWrite.load(), iNewEnd );

			// Now fix the read and write indices
			m_iRead= 0; // we now read from the beginning
			m_iWrite= m_items.size()-1; // point to the first empty entry

			m_items.swap( newItems ); // replace our old FIFO with the new one
		} // end grow()

	}; // end class Queue<>

} // end namespace Atomic

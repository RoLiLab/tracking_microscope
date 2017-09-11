///////////////////////////////////////////////////////////
//
// Queue_lockfree_tests.cpp: Unit tests for lockfree queue
// Copyright (c) 2013 - Kane Lab.  All Rights Reserved
//
// Jeremy Todd
//
//////////////////////////////////////////////////////////////////////////////
#include "Base/base.h"
#include "Queue_lockfree.h"
#include <gtest/gtest.h>


TEST(LockFree_Queue,BasicTests) {

	// Create a non-growable queue, make sure it fails on push-to-full, verify contents
	{	LockFree::Queue<int> q( 3, false );
		EXPECT_TRUE( q.Push(3) );
		EXPECT_TRUE( q.Push(204) );
		EXPECT_TRUE( q.Push(17) );
		EXPECT_FALSE( q.Push(19) );
		int v= 0;
		EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 3, v );
		EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 204, v );
		EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 17, v );
		EXPECT_FALSE( q.Pop(v) );

		// Try again from a different starting index
		EXPECT_TRUE( q.Push(3) );
		EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 3, v );
		EXPECT_TRUE( q.Push(3) );
		EXPECT_TRUE( q.Push(204) );
		EXPECT_TRUE( q.Push(17) );
		EXPECT_FALSE( q.Push(19) );
		EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 3, v );
		EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 204, v );
		EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 17, v );
		EXPECT_FALSE( q.Pop(v) );
	} // end non-growable


	// Check for growing from length N, try this for several values of N and several starting indices
	for( size_t N=1; N<10; ++N ) {
		LockFree::Queue<int> q( N );
		for( size_t iStart=0; iStart<=N; ++iStart ) {

			// Push/pop to advance to our starting index
			for( size_t i=0; i<iStart; ++i ) {
				EXPECT_TRUE( q.Push(1) );
				int v= 0;
				EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( 1, v );
			} // end advancing starting index

			// Now push 2N-1 items, make sure we grow and preserve contents correctly
			for( size_t i=0; i<2*N-1; ++i ) {
				EXPECT_TRUE( q.Push(int(i+15)) );
			} // end pushing
			for( auto i : make_range(2*N-1) ) {
				int v= 0;
				EXPECT_TRUE( q.Pop(v) ); EXPECT_EQ( int(i+15), v );
			} // end pushing
		} // end for starting index
	} // end for N

} // end basic tests

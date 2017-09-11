/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */


#ifndef _VectorAdd_KERNEL_H_
#define _VectorAdd_KERNEL_H_

#include "CUDA_Kernels.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}


// Device code
extern "C" void VecAdd_kernelGPU(
    float *d_Src1,
    float *d_Src2,
	float *d_Dst,
    int numElements
)
{
    // Launch the Vector Add CUDA Kernel
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	//cudaSetDevice(GPU0);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_Src1, d_Src2, d_Dst, numElements);
    //err = cudaGetLastError();
}
 


#endif
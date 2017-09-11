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

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include "CUDA_Kernels.h"

// C = (A + B)/2
__global__ void vectorMean(const int *A, const int *B, int *C, int numElements, int _step)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {	
        int ax = A[i]%_step;
		int ay = A[i]/_step;
	    int bx = B[i]%_step;
		int by = B[i]/_step;
		int x = (ax + bx)*0.5;
		int y = (ay + by)*0.5;
		C[i] = x + y*_step;
    }
}

__global__ void offsetShift(int *A, int numElements, int _step)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		int x = (A[i] % _step) -9;
		int y = (A[i] / _step) -9;
		if (x < 1) x = 1;
		if (y < 1) y = 1;
		A[i] = x + y*_step;
	}
}

__global__ void ind2sub(const int *in, int *out, int _step, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        out[2*i] = in[i]%_step;
		out[2*i+1] = in[i]/_step;
    }
}

__global__ void min_reduceInd(
	const float* const d_array, const int * const d_Indarray, 
	float* d_max, int* item, 
	const size_t elements, const size_t width, 
	int RstCount, int * RstIdx, int * RstRadius,	
	const int * RefInd_pow2, const int * RefInd_y
	)
{
	// position indices for masking-out (Fish yolk ans eye1 position)
	
    __shared__ float shared[threadsPerBlock];
	__shared__ float sharedIdx[threadsPerBlock];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = 0; 
	
    while (gid < elements) {
		// add conditions for the geometry;
		bool InsideRegion = true;
		int gain = 1;		
		if (RstIdx != NULL) {
			int id_x = gid%width;
			int id_y = gid/width;
			if (!(15 < id_x && id_x < 625 && 15 < id_y && id_y < 465))
				//if (!(220 < id_x && id_x < 420 && 140 < id_y && id_y < 340))
				InsideRegion = false;
			if (InsideRegion && d_Indarray == NULL) {
				for (int i = 0; i < RstCount; i++) { // physical boundary (Yolk-eye / eye-eye) - masking out
					int dx=0; int dy=0;
					if (id_x > RstIdx[2*i]) dx = id_x - RstIdx[2*i]; else dx = RstIdx[2*i] - id_x;
					if (id_y > RstIdx[2*i+1]) dy = id_y - RstIdx[2*i+1]; else dy = RstIdx[2*i+1] - id_y;
					int dist = RefInd_pow2[dx] + RefInd_pow2[dy];
					if (dist < RstRadius[i] || dist > 4*RstRadius[i]) {
						InsideRegion = false;
					}
				}
				if (InsideRegion && RstCount > 0 && RstIdx[6] > 0 && RstIdx[7] > 0) { // gain set-up from previous brain position (for eye1 and eye2)
					int dx=0; int dy=0;
					if (id_x > RstIdx[6]) dx = id_x - RstIdx[6]; else dx = RstIdx[6] - id_x;
					if (id_y > RstIdx[7]) dy = id_y - RstIdx[7]; else dy = RstIdx[7] - id_y;
					int dist = RefInd_pow2[dx] + RefInd_pow2[dy];
					if (dist < RstRadius[2])
						gain = 3;
				}
			}
		}
		if (InsideRegion && (shared[tid] > gain*d_array[gid])) {
			// first step of reduction
			shared[tid] = gain*d_array[gid];
			if (d_Indarray == NULL)
				sharedIdx[tid] = gid;
			else
				sharedIdx[tid] = d_Indarray[gid];
		}
        //shared[tid] = min(shared[tid], d_array[gid]);
        gid += gridDim.x*blockDim.x;
    }

    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements) {
			if (shared[tid] > shared[tid + s]) {
				shared[tid] = shared[tid + s];
				sharedIdx[tid] = sharedIdx[tid + s];
			}
            //shared[tid] = min(shared[tid], shared[tid + s]);
			
		}
        __syncthreads();
    }

    if (tid == 0) {
		item[blockIdx.x] = sharedIdx[0];
		d_max[blockIdx.x] = shared[0];
		//atomicMaxf(d_max, shared[0]);		
	}
}

__global__ void min_reduceInd_NoGain(
	const float* const d_array, const int * const d_Indarray, 
	float* d_max, int* item, 
	const size_t elements, const size_t width, 
	int RstCount, int * RstIdx, int * RstRadius,	
	const int * RefInd_pow2, const int * RefInd_y
	)
{
	// position indices for masking-out (Fish yolk ans eye1 position)
	
    __shared__ float shared[threadsPerBlock];
	__shared__ float sharedIdx[threadsPerBlock];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = 0; 
	
    while (gid < elements) {
		// add conditions for the geometry;
		bool InsideRegion = true;
		int gain = 1;		
		if (RstIdx != NULL) {
			int id_x = gid%width;
			int id_y = gid/width;
			if (!(15 < id_x && id_x < 625 && 15 < id_y && id_y < 465))
				InsideRegion = false;
			if (InsideRegion && d_Indarray == NULL) {
				for (int i = 0; i < RstCount; i++) { // physical boundary (Yolk-eye / eye-eye)
					int dx=0; int dy=0;
					if (id_x > RstIdx[2*i]) dx = id_x - RstIdx[2*i]; else dx = RstIdx[2*i] - id_x;
					if (id_y > RstIdx[2*i+1]) dy = id_y - RstIdx[2*i+1]; else dy = RstIdx[2*i+1] - id_y;
					int dist = RefInd_pow2[dx] + RefInd_pow2[dy];
					if (dist < RstRadius[i] || dist > 4*RstRadius[i]) {
						InsideRegion = false;
					}
				}
				//if (InsideRegion && RstCount > 0 && RstIdx[6] > 0 && RstIdx[7] > 0) { // gain set-up from previous brain position (for eye1 and eye2)
				//	int dx=0; int dy=0;
				//	if (id_x > RstIdx[6]) dx = id_x - RstIdx[6]; else dx = RstIdx[6] - id_x;
				//	if (id_y > RstIdx[7]) dy = id_y - RstIdx[7]; else dy = RstIdx[7] - id_y;
				//	int dist = RefInd_pow2[dx] + RefInd_pow2[dy];
				//	if (dist < RstRadius[2]) gain = 3;
				//}
			}
		}
		if (InsideRegion && (shared[tid] > gain*d_array[gid])) {
			// first step of reduction
			shared[tid] = gain*d_array[gid];
			if (d_Indarray == NULL)
				sharedIdx[tid] = gid;
			else
				sharedIdx[tid] = d_Indarray[gid];
		}
        //shared[tid] = min(shared[tid], d_array[gid]);
        gid += gridDim.x*blockDim.x;
    }

    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements) {
			if (shared[tid] > shared[tid + s]) {
				shared[tid] = shared[tid + s];
				sharedIdx[tid] = sharedIdx[tid + s];
			}
            //shared[tid] = min(shared[tid], shared[tid + s]);
			
		}
        __syncthreads();
    }

    if (tid == 0) {
		item[blockIdx.x] = sharedIdx[0];
		d_max[blockIdx.x] = shared[0];
		//atomicMaxf(d_max, shared[0]);		
	}
}

extern "C" void reductionMaxIdxGPU_NIR(
	float *d_Src, 
	float *d_Dst,	int *item_Dst,
	int width, int height,
	int RstCount, int * RstIdx, int * RstRadius, 
	float *d_Dummy, 	int *item_Dummy, // 1024 vectors
	const int * RefInd_pow2, const int * RefInd_y
)
{	
    int blocksPerGrid =(width*height + threadsPerBlock - 1) / threadsPerBlock;	
	//cudaSetDevice(GPU0);
	min_reduceInd<<<blocksPerGrid, threadsPerBlock>>>(
		(const float *) d_Src, NULL, 
		d_Dummy, item_Dummy,  
		width*height, width, 
		RstCount, RstIdx, RstRadius,
		RefInd_pow2, RefInd_y);	

	int numElement = blocksPerGrid;
	blocksPerGrid =(numElement + threadsPerBlock - 1) / threadsPerBlock;	
	//cudaSetDevice(GPU0);
	min_reduceInd<<<blocksPerGrid, threadsPerBlock>>>(
		(const float *) d_Dummy, item_Dummy, 
		d_Dst, item_Dst, 
		numElement, 
		width, 
		0, NULL, NULL,
		RefInd_pow2, RefInd_y);	
	//cudaSetDevice(GPU0);
	if (RstCount == 2) {
		vectorMean<<<1, 1>>>((const int *) item_Dst-1, (const int *)item_Dst, item_Dst+1, 1, width);	// center = (eye1 + eye2)/2
		vectorMean<<<1, 1>>>((const int *) item_Dst-2, (const int *)item_Dst+1, item_Dst+2, 1, width);	// Fish center = (yolk + center)/2
		//offsetShift<<<1, 4>>>((int *)item_Dst - 2, 4, width);
	}
}


extern "C" void reductionMaxIdxGPU_NIR_NOGain(
	float *d_Src, 
	float *d_Dst,	int *item_Dst,
	int width, int height,
	int RstCount, int * RstIdx, int * RstRadius, 
	float *d_Dummy, 	int *item_Dummy, // 1024 vectors
	const int * RefInd_pow2, const int * RefInd_y
)
{	
    int blocksPerGrid =(width*height + threadsPerBlock - 1) / threadsPerBlock;	
	min_reduceInd_NoGain<<<blocksPerGrid, threadsPerBlock>>>(
		(const float *) d_Src, NULL, 
		d_Dummy, item_Dummy,  
		width*height, width, 
		RstCount, RstIdx, RstRadius, 
		RefInd_pow2, RefInd_y);	

	int numElement = blocksPerGrid;
	blocksPerGrid =(numElement + threadsPerBlock - 1) / threadsPerBlock;	
	//cudaSetDevice(GPU0);
	min_reduceInd<<<blocksPerGrid, threadsPerBlock>>>(
		(const float *) d_Dummy, item_Dummy, 
		d_Dst, item_Dst, 
		numElement, width, 
		0, NULL, NULL,
		RefInd_pow2, RefInd_y);	

	if (RstCount == 2) {
		vectorMean<<<1, 1>>>((const int *) item_Dst-1, (const int *)item_Dst, item_Dst+1, 1, width);	// center = (eye1 + eye2)/2
		vectorMean<<<1, 1>>>((const int *) item_Dst-2, (const int *)item_Dst+1, item_Dst+2, 1, width);	// Fish center = (yolk + center)/2
		//offsetShift<<<1, 4>>>((int *)item_Dst - 2, 4, width);
	}
}


extern "C" void LinearIdx2SubInd( int * Idx, int * Idx_sub, int width, int Count)
{	
	//cudaSetDevice(GPU0);
	ind2sub<<<1, Count>>>((const int *) Idx, Idx_sub, width, Count);	
}

__global__ void threhold(const float * src, int * idx2d, int width, int numElements, float * xpos, float * ypos, float * labelmap)
{
	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	int LT_x = int(idx2d[0]) - int(numElements / 2);
	int LT_y = int(idx2d[1]) - int(numElements / 2);
	float threshold_yolk = src[idx2d[0] + idx2d[1]*width] * 0.9;
	int numElements_2d = numElements*numElements;
	
	while (gid < numElements_2d)
	{
		int xp = LT_x + gid / numElements;
		int yp = LT_y + gid % numElements;
		int src_lnidx = xp + (yp * width);
		xpos[gid] = 0;
		ypos[gid] = 0;		
		labelmap[gid] = 0;
		if (src[src_lnidx] > threshold_yolk) { // check threshold
			xpos[gid] = xp;
			ypos[gid] = yp;
			labelmap[gid] = 1;
		}		
	}
	__syncthreads();
	/*gid = (blockDim.x * blockIdx.x) + tid;  // 1
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s && gid < numElements_2d) {
			xpos[tid] += xpos[tid + s];
			ypos[tid] += ypos[tid + s];
			labelmap[tid] += labelmap[tid + s];
		}
		__syncthreads();
	}
	xpos[0] = xpos[0] + xpos[1] + xpos[2];
	ypos[0] = ypos[0] + ypos[1] + ypos[2];
	labelmap[0] = labelmap[0] + labelmap[1] + labelmap[2];
	*/
}

extern "C" void thresholdimg_GPU(float * src, int * idx2d, int width, float * xpos, float * ypos, float * labelmap)
{
	int count = 50;
	int blocksPerGrid = (count*count + threadsPerBlock - 1) / threadsPerBlock;
	threhold << <blocksPerGrid, threadsPerBlock >> >((const float *)src, idx2d, width, count, xpos, ypos, labelmap);
}
#endif
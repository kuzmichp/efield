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

// Utilities and system includes

#include <helper_cuda.h>
#include <math.h>

#define COULOMBS_CONST 9

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__global__ void
cudaProcess(unsigned int *g_odata, int imgw)
{
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

    uchar4 c4 = make_uchar4((x & 0x20)?100:0,0,(y & 0x20)?100:0,0);
    g_odata[y*imgw+x] = rgbToInt(200, 130, 155);
}

__global__ void
determineIntensity(unsigned int *g_odata, int imgw, int *d_x, int *d_y, int *d_v, int chrg_num, float *v)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

    int i;

    float2 res = make_float2(0, 0);

    for (i = 0; i < chrg_num; i++)
    {
    	int dx = x - d_x[i];
    	int dy = y - d_y[i];

    	float dist2 = (float)(dx*dx + dy*dy);
    	// E = k*Q/r^2
    	float lcl_int = COULOMBS_CONST*d_v[i]/dist2;

    	float dist = sqrtf((float)(dx*dx + dy*dy));

    	res.x += (dx/dist)*lcl_int;
    	res.y += (dy/dist)*lcl_int;
    }

    float gbl_int = sqrtf(res.x*res.x + res.y*res.y);
    *v = gbl_int;

    int clr = (int)gbl_int;

    g_odata[y*imgw+x] = rgbToInt(clr, clr, clr);
}

extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                   unsigned int *g_odata, unsigned int *x_pos,
                   unsigned int *y_pos, int *vals,
                   int imgw, int chrg_num, float *v)
{
	int *d_x, *d_y, *d_v;

	cudaCheckErrors(cudaMalloc((void **)&d_x, sizeof(int) * chrg_num));
	cudaCheckErrors(cudaMalloc((void **)&d_y, sizeof(int) * chrg_num));
	cudaCheckErrors(cudaMalloc((void **)&d_v, sizeof(int) * chrg_num));

	cudaCheckErrors(cudaMemcpy(d_x, x_pos, sizeof(int)*chrg_num, cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMemcpy(d_y, y_pos, sizeof(int)*chrg_num, cudaMemcpyHostToDevice));
	cudaCheckErrors(cudaMemcpy(d_v, vals, sizeof(int)*chrg_num, cudaMemcpyHostToDevice));

	determineIntensity<<<grid, block>>>(g_odata, imgw, d_x, d_y, d_v, chrg_num, v);

	if (cudaSuccess != cudaGetLastError())
	{
	    printf("Kernel error!\n");
	    exit(EXIT_FAILURE);
	}

	fprintf(stderr, "Global intensity: %f\n", *v);
}

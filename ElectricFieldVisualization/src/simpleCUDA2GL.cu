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

//9e9
#define COULOMBS_CONST 9

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

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
determineIntensity(unsigned int *g_odata, int imgw, int *d_x, int *d_y, int *d_v, int chrg_num)
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
    	//TODO: rozpatrywać jedynie ładunki o określonym położeniu
    	int dx = x - d_x[i];
    	int dy = y - d_y[i];

    	int dist2 = dx*dx + dy*dy;
    	// E = k*Q/r^2 (lcl_int e9)
    	float lcl_int = (COULOMBS_CONST*d_v[i]/(float)dist2);
    	//intens[i] = lcl_int;
    	float dist = sqrtf((float)dist2);

    	res.x += (dx/dist)*lcl_int;
    	res.y += (dy/dist)*lcl_int;
    }

    res.x *= 10000;
    res.y *= 10000;

    float gbl_int = sqrtf(res.x*res.x + res.y*res.y);
    int clr = (int)gbl_int;

    g_odata[y*imgw+x] = rgbToInt(clr/10, clr/15, clr/3);
}

__global__ void
moveCharges(int *d_x, int *d_y, int *d_dir, int chrg_num)
{
    int tid = threadIdx.x;

    if (tid < chrg_num)

    if (d_dir[tid] == -1)
    {
    	if (d_y[tid] <= 0) d_dir[tid] *= -1;
    	else d_y[tid] -= 1;
    }
    else if (d_dir[tid] == 1)
    {
    	if (d_y[tid] >= 511) d_dir[tid] *= -1;
    	else d_y[tid] += 1;
    }
    else if (d_dir[tid] == -2)
    {
    	if (d_x[tid] <= 0) d_dir[tid] *= -1;
    	else d_x[tid] -= 1;
    }
    else if (d_dir[tid] == 2)
    {
    	if (d_x[tid] >= 511) d_dir[tid] *= -1;
    	else d_x[tid] += 1;
    }
    else {}

	__syncthreads();
}

extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                   unsigned int *g_odata, int *x_pos,
                   int *y_pos, int *dir, int *vals,
                   int imgw, int chrg_num, int mv)
{
	int i;
	// device vectors
	int *d_x, *d_y, *d_v, *d_dir;
	dim3 mvBlock(chrg_num, 1, 1);

	fprintf(stderr, "Kierunki:\n");

	for (i = 0; i < chrg_num; i++)
	{
		fprintf(stderr, "[%d] - %d\n", i + 1, dir[i]);
	}

	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_x, sizeof(int) * chrg_num));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_y, sizeof(int) * chrg_num));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_v, sizeof(int) * chrg_num));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_dir, sizeof(int) * chrg_num));

	CUDA_CHECK_RETURN(cudaMemcpy((void *)d_x, (void *)x_pos, sizeof(int)*chrg_num, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy((void *)d_y, (void *)y_pos, sizeof(int)*chrg_num, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy((void *)d_v, (void *)vals, sizeof(int)*chrg_num, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy((void *)d_dir, (void *)dir, sizeof(int)*chrg_num, cudaMemcpyHostToDevice));

	if (mv == 0) moveCharges<<<1, mvBlock>>>(d_x, d_y, d_dir, chrg_num);

	determineIntensity<<<grid, block>>>(g_odata, imgw, d_x, d_y, d_v, chrg_num);

	if (cudaSuccess != cudaGetLastError())
	{
	    printf("Kernel error!\n");
	    exit(EXIT_FAILURE);
	}

	//update positions
	if (1 == 1)
	{
		CUDA_CHECK_RETURN(cudaMemcpy((void *)x_pos, (void *)d_x, sizeof(int)*chrg_num, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy((void *)y_pos, (void *)d_y, sizeof(int)*chrg_num, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy((void *)dir, (void *)d_dir, sizeof(int)*chrg_num, cudaMemcpyDeviceToHost));
	}
}

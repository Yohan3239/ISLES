#ifndef CUDAFUNCTIONS_H
#define CUDAFUNCTIONS_H

#ifndef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

    // Function declaration (host function that calls the CUDA kernel)
bool convolveHost(
    float* input, int inputNum, int depth, int height, int width,
    float* filters, int outputNum, int filterDepth, int filterHeight, int filterWidth, // skip inputNum bc same
    float* output, int outputDepth, int outputHeight, int outputWidth, // skip outnputNum bc same
    float* bias,
    int stride, int padding);

bool calcInputGradHost(
    float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultInputGrad, // same dim as input
    int padding,
    int stride);

bool calcFilterGradHost(
    float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultGrad, // same dim as input
    int padding,
    int stride);
#endif


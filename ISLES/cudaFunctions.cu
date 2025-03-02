#include <cuda_runtime.h>
#include "cudaFunctions.h"
#include <iostream>
#include <vector>

using namespace std;

__global__ void convolveKernel(
    float* input, int inputNum, int depth, int height, int width,
    float* filters, int outputNum, int filterDepth, int filterHeight, int filterWidth,
    float* output, int outputDepth, int outputHeight, int outputWidth,
    float* bias, 
    int stride, int padding
) {
    int outChannel = blockIdx.x;
    int z = blockIdx.y * blockDim.z + threadIdx.z;
    int y = blockIdx.z * blockDim.y + threadIdx.y;
    int x = threadIdx.x;

    float sum = 0.f;
    for (int inChannel = 0; inChannel < inputNum; ++inChannel) {
        for (int fz = 0; fz < filterDepth; ++fz) {
            for (int fy = 0; fy < filterHeight; ++fy) {
                for (int fx = 0; fx < filterWidth; ++fx) {
                    // Find the corresponding position in the input grid
                    int inputZ = z * stride + fz - padding;  // Apply stride and padding
                    int inputY = y * stride + fy - padding;
                    int inputX = x * stride + fx - padding;

                    // Only compute if within bounds
                    if (inputZ >= 0 && inputZ < depth && inputY >= 0 && inputY < height && inputX >= 0 && inputX < width) {
                        // Multiply and sum the product of filter and input value
                        
                        sum += input[((inChannel * depth + inputZ) * height + inputY) * width + inputX] * filters[(((outChannel * inputNum + inChannel) * filterDepth + fz) * filterHeight + fy) * filterWidth + fx];
                    }
                }
            }
        }
    }
    output[((outChannel * outputDepth + z) * outputHeight + y) * outputWidth + x] = sum + bias[outChannel];
}

__global__ void calcInputGradKernel(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultInputGrad, // same dim as input
    int padding,
    int stride) {

    int ic = blockIdx.x;
    int z = blockIdx.y * blockDim.z + threadIdx.z;
    int y = blockIdx.z * blockDim.y + threadIdx.y;
    int x = threadIdx.x;

    int gz_min = max(0, int(ceilf((float)(z - origFilterDepth + 1) / stride)));
    int gz_max = min(inputDepth - 1, z / stride);

    int gy_min = max(0, int(ceilf((float)(y - origFilterHeight + 1) / stride)));
    int gy_max = min(inputHeight - 1, y / stride);

    int gx_min = max(0, int(ceilf((float)(x - origFilterWidth + 1) / stride)));
    int gx_max = min(inputWidth - 1, x / stride);

    float sum = 0.f;
    for (int oc = 0; oc < origFilterOutNum; ++oc) {
        // Loop over each element in the filter spatially.
        for (int fz = 0; fz < origFilterDepth; ++fz) {
            for (int fy = 0; fy < origFilterHeight; ++fy) {
                for (int fx = 0; fx < origFilterWidth; ++fx) {
                    // In the forward pass (with same padding), the input voxel (z, y, x)
                    // is affected by an output voxel at:
                    // out_z = (z + padding - fz) / stride, provided (z + padding - fz) is divisible by stride,
                    // and similarly for y and x.
                    int out_z = z + padding - fz;
                    int out_y = y + padding - fy;
                    int out_x = x + padding - fx;

                    // Check if these coordinates align with the stride.
                    if ((out_z % stride == 0) && (out_y % stride == 0) && (out_x % stride == 0)) {
                        out_z /= stride;
                        out_y /= stride;
                        out_x /= stride;

                        // For same padding, we assume the output dims equal the input dims.
                        if (out_z >= 0 && out_z < inputDepth &&
                            out_y >= 0 && out_y < inputHeight &&
                            out_x >= 0 && out_x < inputWidth)
                        {
                            int inputSlice = inputHeight * inputWidth;
                            int filterVol = origFilterDepth * origFilterHeight * origFilterWidth;

                            sum += lossPrevGrad[oc * inputDepth * inputSlice + out_z * inputSlice + out_y * inputWidth + out_x] *
                                rotatedFilters[oc * (origFilterInNum * filterVol) + ic * filterVol + fz * (origFilterHeight * origFilterWidth) + fy * origFilterWidth + fx];
                        }
                    }
                }
            }
        }
    }

    // Write the accumulated sum into the result for input voxel (ic, z, y, x).
    int inputVol = inputDepth * inputHeight * inputWidth;
    resultInputGrad[ic * inputVol + z * (inputHeight * inputWidth) + y * inputWidth + x] = sum;

   
}

__global__ void calcFilterGradKernel(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultGrad, // same dim as input
    int padding,
    int stride) {


    
    int oc = blockIdx.x;
    int ic = blockIdx.y;

    int totalThreads = origFilterHeight * origFilterWidth;
    int nBlocksPerDepth = ceil(float(totalThreads) / 1024);
    int depthIndex = blockIdx.z / nBlocksPerDepth; 
    int blockIdxWithinDepth = blockIdx.z % nBlocksPerDepth; 

    int threadID = threadIdx.y * blockDim.x + threadIdx.x; 
    int filterElementIndex = blockIdxWithinDepth * (blockDim.x * blockDim.y) + threadID;

    if (filterElementIndex >= totalThreads) {
        return;
    }

    int z = depthIndex;
    int y = filterElementIndex / origFilterWidth;  
    int x = filterElementIndex % origFilterWidth;  

    

    float sum = 0.f;

    for (int gz = 0; gz < inputDepth; ++gz) {
        for (int gy = 0; gy < inputHeight; ++gy) {
            for (int gx = 0; gx < inputWidth; ++gx) {
                int zCord = stride * gz + z - padding;
                int yCord = stride * gy + y - padding;
                int xCord = stride * gx + x - padding;
                if (zCord >= 0 && zCord < inputDepth &&
                    yCord >= 0 && yCord < inputHeight &&
                    xCord >= 0 && xCord < inputWidth) {
                    sum += lossPrevGrad[oc * inputDepth * inputHeight * inputWidth + gz * inputHeight * inputWidth + gy * inputWidth + gx] * input[ic * inputDepth * inputHeight * inputWidth + zCord * inputHeight * inputWidth + yCord * inputWidth + xCord];
                } // gonna assume same-padding bc otherwise overcomplicated


            }
        }
    }
    resultGrad[oc * (origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth) + ic * (origFilterDepth * origFilterHeight * origFilterWidth) + z * (origFilterHeight * origFilterWidth) + y * origFilterWidth + x] = sum;
}



void convolveHost(
    float* input, int inputNum, int depth, int height, int width, 
    float* filters, int outputNum, int filterDepth, int filterHeight, int filterWidth, // skip inputNum bc same
    float* output, int outputDepth, int outputHeight, int outputWidth, // skip outnputNum bc same
    float* bias, 
    int stride, int padding
) {

    int inputAllocSize = inputNum * depth * height * width * sizeof(float);
    int filterAllocSize = outputNum * inputNum * filterDepth * filterHeight * filterWidth * sizeof(float);
    int outputAllocSize = outputNum * outputDepth * outputHeight * outputWidth * sizeof(float);
    int biasAllocSize = outputNum * sizeof(float);

    dim3 gridDim(outputNum, outputDepth, outputHeight);
    dim3 blockDim(outputWidth, 1, 1);

    float* inputHostPinned = nullptr; // must initialise or segmentation fault ...
    float* filterHostPinned = nullptr;
    float* outputHostPinned = nullptr;
    float* biasHostPinned = nullptr;

    float* inputDeviceData;
    float* filterDeviceData;
    float* outputDeviceData;
    float* biasDeviceData;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocHost((void**)&inputHostPinned, inputAllocSize);
    cudaMallocHost((void**)&filterHostPinned, filterAllocSize);
    cudaMallocHost((void**)&outputHostPinned, outputAllocSize);
    cudaMallocHost((void**)&biasHostPinned, biasAllocSize);

    cudaMalloc(&biasDeviceData, biasAllocSize);
    cudaMalloc(&inputDeviceData, inputAllocSize);
    cudaMalloc(&filterDeviceData, filterAllocSize);
    cudaMalloc(&outputDeviceData, outputAllocSize);
    //data transfers
    memcpy(inputHostPinned, input, inputAllocSize);
    memcpy(filterHostPinned, filters, filterAllocSize);
    memcpy(biasHostPinned, bias, biasAllocSize);
    
    cudaMemcpyAsync(inputDeviceData, inputHostPinned, inputAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(filterDeviceData, filterHostPinned, filterAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(biasDeviceData, biasHostPinned, biasAllocSize, cudaMemcpyHostToDevice, stream);

    convolveKernel<<<gridDim, blockDim>>>(inputDeviceData, inputNum, depth, height, width,
        filterDeviceData, outputNum, filterDepth, filterHeight, filterWidth, // skip inputNum bc same
        outputDeviceData, outputDepth, outputHeight, outputWidth, // skip outnputNum bc same
        biasDeviceData,
        stride, padding);
    
    cudaMemcpyAsync(outputHostPinned, outputDeviceData, outputAllocSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    memcpy(output, outputHostPinned, outputAllocSize);


    cudaFree(inputDeviceData);
    cudaFree(filterDeviceData);
    cudaFree(outputDeviceData);
    cudaFree(biasDeviceData);

    cudaFreeHost(inputHostPinned);
    cudaFreeHost(filterHostPinned);
    cudaFreeHost(outputHostPinned);
    cudaFreeHost(biasHostPinned);

    cudaStreamDestroy(stream);
}


void calcInputGradHost(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultInputGrad, // same dim as input
    int padding,
    int stride) {

    int inputAllocSize = origFilterInNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    int filterAllocSize = origFilterOutNum * origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth * sizeof(float);
    int lossPrevGradAllocSize = origFilterOutNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    int resultInputGradAllocSize = inputAllocSize; // exactly same size bc ye

    dim3 gridDim(origFilterInNum, inputDepth, inputHeight);
    dim3 blockDim(inputWidth, 1, 1);

    float* inputHostPinned = nullptr;
    float* filterHostPinned = nullptr;
    float* lossPrevGradHostPinned = nullptr;
    float* resultInputGradHostPinned = nullptr;

    float* inputDeviceData;
    float* filterDeviceData;
    float* lossPrevGradDeviceData;
    float* resultInputGradDeviceData;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocHost((void**)&inputHostPinned, inputAllocSize);
    cudaMallocHost((void**)&filterHostPinned, filterAllocSize);
    cudaMallocHost((void**)&lossPrevGradHostPinned, lossPrevGradAllocSize);
    cudaMallocHost((void**)&resultInputGradHostPinned, resultInputGradAllocSize);

    cudaMalloc(&inputDeviceData, inputAllocSize);
    cudaMalloc(&filterDeviceData, filterAllocSize);
    cudaMalloc(&lossPrevGradDeviceData, lossPrevGradAllocSize);
    cudaMalloc(&resultInputGradDeviceData, resultInputGradAllocSize);

    memcpy(inputHostPinned, input, inputAllocSize);
    memcpy(filterHostPinned, rotatedFilters, filterAllocSize);
    memcpy(lossPrevGradHostPinned, lossPrevGrad, lossPrevGradAllocSize);

    cudaMemcpyAsync(inputDeviceData, inputHostPinned, inputAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(filterDeviceData, filterHostPinned, filterAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(lossPrevGradDeviceData, lossPrevGradHostPinned, lossPrevGradAllocSize, cudaMemcpyHostToDevice, stream);

    calcInputGradKernel<<<gridDim, blockDim, 0, stream>>>(inputDeviceData, origFilterInNum, inputDepth, inputHeight, inputWidth,
        lossPrevGradDeviceData, origFilterOutNum, //same dim as input bc same padding
        filterDeviceData, origFilterDepth, origFilterHeight, origFilterWidth,
        resultInputGradDeviceData, // same dim as input
        padding,
        stride);

    cudaMemcpyAsync(resultInputGradHostPinned, resultInputGradDeviceData, resultInputGradAllocSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    memcpy(resultInputGrad, resultInputGradHostPinned, resultInputGradAllocSize);

    cudaFree(inputDeviceData);
    cudaFree(filterDeviceData);
    cudaFree(lossPrevGradDeviceData);
    cudaFree(resultInputGradDeviceData);

    cudaFreeHost(inputHostPinned);
    cudaFreeHost(filterHostPinned);
    cudaFreeHost(lossPrevGradHostPinned);
    cudaFreeHost(resultInputGradHostPinned);

    cudaStreamDestroy(stream);

}

void calcFilterGradHost(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultGrad, // same dim as input
    int padding,
    int stride) {

    unsigned long inputAllocSize = origFilterInNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    unsigned long filterAllocSize = origFilterOutNum * origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth * sizeof(float);
    unsigned long lossPrevGradAllocSize = origFilterOutNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    unsigned long resultFilterGradAllocSize = filterAllocSize; // exactly same size bc ye

    int totalThreads = origFilterHeight * origFilterWidth;
    int maxThreads = min(1024, totalThreads);
    int blockH = min(origFilterHeight, maxThreads);
    int blockW = max(1, maxThreads / blockH);

    dim3 blockDim(blockH, blockW, 1);
    dim3 gridDim(origFilterOutNum, origFilterInNum, origFilterDepth * ceil(float(totalThreads) / 1024));

    float* inputHostPinned = nullptr;
    float* filterHostPinned = nullptr;
    float* lossPrevGradHostPinned = nullptr;
    float* resultFilterGradHostPinned = nullptr;

    float* inputDeviceData;
    float* filterDeviceData;
    float* lossPrevGradDeviceData;
    float* resultFilterGradDeviceData;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocHost((void**)&inputHostPinned, inputAllocSize);
    cudaMallocHost((void**)&filterHostPinned, filterAllocSize);
    cudaMallocHost((void**)&lossPrevGradHostPinned, lossPrevGradAllocSize);
    cudaMallocHost((void**)&resultFilterGradHostPinned, resultFilterGradAllocSize);

    
    cudaMalloc(&inputDeviceData, inputAllocSize);
    cudaMalloc(&filterDeviceData, filterAllocSize);
    cudaMalloc(&lossPrevGradDeviceData, lossPrevGradAllocSize);
    cudaMalloc(&resultFilterGradDeviceData, resultFilterGradAllocSize);

   
    memcpy(inputHostPinned, input, inputAllocSize);
    memcpy(filterHostPinned, rotatedFilters, filterAllocSize);
    memcpy(lossPrevGradHostPinned, lossPrevGrad, lossPrevGradAllocSize);

    cudaMemcpyAsync(inputDeviceData, inputHostPinned, inputAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(filterDeviceData, filterHostPinned, filterAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(lossPrevGradDeviceData, lossPrevGradHostPinned, lossPrevGradAllocSize, cudaMemcpyHostToDevice, stream);

    calcFilterGradKernel<<<gridDim, blockDim, 0, stream >>> (inputDeviceData, origFilterInNum, inputDepth, inputHeight, inputWidth,
        lossPrevGradDeviceData, origFilterOutNum, //same dim as input bc same padding
        filterDeviceData, origFilterDepth, origFilterHeight, origFilterWidth,
        resultFilterGradDeviceData, // same dim as input
        padding,
        stride);

    cudaMemcpyAsync(resultFilterGradHostPinned, resultFilterGradDeviceData, resultFilterGradAllocSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    memcpy(resultGrad, resultFilterGradHostPinned, resultFilterGradAllocSize);

    cudaFree(inputDeviceData);
    cudaFree(filterDeviceData);
    cudaFree(lossPrevGradDeviceData);
    cudaFree(resultFilterGradDeviceData);

    cudaFreeHost(inputHostPinned);
    cudaFreeHost(filterHostPinned);
    cudaFreeHost(lossPrevGradHostPinned);
    cudaFreeHost(resultFilterGradHostPinned);

    cudaStreamDestroy(stream);

}
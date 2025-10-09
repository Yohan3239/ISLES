#include <cuda_runtime.h>
#include "cudaFunctions.h"
#include <iostream>
#include <vector>

using namespace std;

__global__ void static convolveKernel(
    float* input, int inputNum, int depth, int height, int width,
    float* filters, int outputNum, int filterDepth, int filterHeight, int filterWidth,
    float* output, int outputDepth, int outputHeight, int outputWidth,
    float* bias, 
    int stride, int padding
) {
    // Calculating global coordinates of threads
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
    // Atomic Add to prevent race conditions
    atomicAdd(&output[((outChannel * outputDepth + z) * outputHeight + y) * outputWidth + x], sum + bias[outChannel]);
}

__global__ void static calcInputGradKernel(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultInputGrad, // same dim as input
    int padding,
    int stride) {
    // Calculating global coordinate of threads
    int ic = blockIdx.x;
    int z = blockIdx.y * blockDim.z + threadIdx.z;
    int y = blockIdx.z * blockDim.y + threadIdx.y;
    int x = threadIdx.x;

    float sum = 0.f;
    for (int oc = 0; oc < origFilterOutNum; ++oc) {
        // Loop over each element in filters
        for (int fz = 0; fz < origFilterDepth; ++fz) {
            for (int fy = 0; fy < origFilterHeight; ++fy) {
                for (int fx = 0; fx < origFilterWidth; ++fx) {
                    // out_z = (z + padding - fz) / stride, similarly for x and y.
                    int out_z = z + padding - fz; 
                    int out_y = y + padding - fy;
                    int out_x = x + padding - fx;

                    // Check if these coordinates align with the stride.
                    if ((out_z % stride == 0) && (out_y % stride == 0) && (out_x % stride == 0)) {
                        out_z /= stride; // Undoing the z * stride - padding + fz
                        out_y /= stride;
                        out_x /= stride;

                        // For same padding, assume the output dim = input dim
                        if (out_z >= 0 && out_z < inputDepth &&
                            out_y >= 0 && out_y < inputHeight &&
                            out_x >= 0 && out_x < inputWidth)
                        {
                            int inputSlice = inputHeight * inputWidth; // Values calculated once so no need to do it twice in indexing
                            int filterVol = origFilterDepth * origFilterHeight * origFilterWidth;

                            // Dot product of output gradient with the rotated filters
                            sum += lossPrevGrad[oc * inputDepth * inputSlice + out_z * inputSlice + out_y * inputWidth + out_x] *
                                rotatedFilters[oc * (origFilterInNum * filterVol) + ic * filterVol + fz * (origFilterHeight * origFilterWidth) + fy * origFilterWidth + fx];
                        }
                    }
                }
            }
        }
    }

    // Write sum into tensor
    int inputVol = inputDepth * inputHeight * inputWidth;
    // Atomic add to prevent race conditions
    atomicAdd(&resultInputGrad[ic * inputVol + z * inputHeight * inputWidth + y * inputWidth + x], sum);
}

__global__ void static calcFilterGradKernel(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultGrad, // same dim as input
    int padding,
    int stride) {
    
    int oc = blockIdx.x;
    int ic = blockIdx.y;

    int totalThreads = origFilterHeight * origFilterWidth;
    int blocksPerDepth = ceil(float(totalThreads) / 1024); // How many blocks there should be for one depth representation
    int depthIndex = blockIdx.z / blocksPerDepth;
    int blockIdxInDepth = blockIdx.z % blocksPerDepth; // Which number the block is in the depth group

    int threadIDX = threadIdx.y * blockDim.x + threadIdx.x; // 1D index of the thread computed from the filter slice
    int filterIdx = blockIdxInDepth * (blockDim.x * blockDim.y) + threadIDX; // index of which filter weight the thread is processing

    if (filterIdx >= totalThreads) { // return if filterIdx is too large
        return;
    }

    int z = depthIndex;
    int y = filterIdx / origFilterWidth;
    int x = filterIdx % origFilterWidth;

    

    float sum = 0.f;

    for (int gz = 0; gz < inputDepth; ++gz) {
        for (int gy = 0; gy < inputHeight; ++gy) {
            for (int gx = 0; gx < inputWidth; ++gx) {
                int zCord = stride * gz + z - padding;
                int yCord = stride * gy + y - padding;
                int xCord = stride * gx + x - padding;
                if (zCord >= 0 && zCord < inputDepth && // 
                    yCord >= 0 && yCord < inputHeight &&
                    xCord >= 0 && xCord < inputWidth) {
                    // Dot products of gradient with input
                    sum += lossPrevGrad[oc * inputDepth * inputHeight * inputWidth + gz * inputHeight * inputWidth + gy * inputWidth + gx] * input[ic * inputDepth * inputHeight * inputWidth + zCord * inputHeight * inputWidth + yCord * inputWidth + xCord];
                } 


            }
        }
    }
    // Again, atomic add
    atomicAdd(&resultGrad[oc * (origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth) 
        + ic * (origFilterDepth * origFilterHeight * origFilterWidth) 
        + z * (origFilterHeight * origFilterWidth) 
        + y * origFilterWidth + x], sum);
}



bool convolveHost(
    float* input, int inputNum, int depth, int height, int width, 
    float* filters, int outputNum, int filterDepth, int filterHeight, int filterWidth, // skip inputNum bc same
    float* output, int outputDepth, int outputHeight, int outputWidth, // skip outnputNum bc same
    float* bias, 
    int stride, int padding
) {
 
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem); // Used for debugging
    //cout << "Free Memory before allocation: " << freeMem << " / " << totalMem << " bytes\n";
    size_t inputAllocSize = inputNum * depth * height * width * sizeof(float); // Amount of memory to be allocated
    size_t filterAllocSize = outputNum * inputNum * filterDepth * filterHeight * filterWidth * sizeof(float); 
    size_t outputAllocSize = outputNum * outputDepth * outputHeight * outputWidth * sizeof(float);
    size_t biasAllocSize = outputNum * sizeof(float);

    dim3 gridDim(outputNum, outputDepth, outputHeight); // oc/depth/height
    dim3 blockDim(outputWidth, 1, 1); // width


    float* inputHostPinned = nullptr;  // Pinned memory of host (CPU)
    float* filterHostPinned = nullptr;// must initialise or segmentation fault ...
    float* outputHostPinned = nullptr;
    float* biasHostPinned = nullptr;

    float* inputDeviceData; // Device data (GPU data)
    float* filterDeviceData;
    float* outputDeviceData;
    float* biasDeviceData;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaError_t mallocError;
    // Allocate memory to host with corresponding size
    mallocError = cudaMallocHost((void**)&inputHostPinned, inputAllocSize); 
    // Check for error in memory allocation and return
    if (mallocError != cudaSuccess) { 
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&filterHostPinned, filterAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&outputHostPinned, outputAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&biasHostPinned, biasAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }

    mallocError = cudaMalloc(&biasDeviceData, biasAllocSize); // Allocate memory to device with corresponding size
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&inputDeviceData, inputAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&filterDeviceData, filterAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&outputDeviceData, outputAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }

    // Data copy from input to host memory
    memcpy(inputHostPinned, input, inputAllocSize);
    memcpy(filterHostPinned, filters, filterAllocSize);
    memcpy(biasHostPinned, bias, biasAllocSize);
    
    // Data copy from host memory to device memory
    cudaMemcpyAsync(inputDeviceData, inputHostPinned, inputAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(filterDeviceData, filterHostPinned, filterAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(biasDeviceData, biasHostPinned, biasAllocSize, cudaMemcpyHostToDevice, stream);

    // Run main kernel
    convolveKernel<<<gridDim, blockDim>>>(inputDeviceData, inputNum, depth, height, width,
        filterDeviceData, outputNum, filterDepth, filterHeight, filterWidth, // skip inputNum bc same
        outputDeviceData, outputDepth, outputHeight, outputWidth, // skip outnputNum bc same
        biasDeviceData,
        stride, padding);
    
    // Error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "Error: CUDA kernel failed: " << cudaGetErrorString(error) << "\n";
        return false;
    }

    // Sync all device threads before copying
    cudaDeviceSynchronize();
    // Copy from device to host
    cudaMemcpyAsync(outputHostPinned, outputDeviceData, outputAllocSize, cudaMemcpyDeviceToHost, stream);
    // Sync stream
    cudaStreamSynchronize(stream);

    // Copy from host to output
    memcpy(output, outputHostPinned, outputAllocSize);

    // Free allocated memory
    cudaFree(inputDeviceData);
    cudaFree(filterDeviceData);
    cudaFree(outputDeviceData);
    cudaFree(biasDeviceData);

    cudaFreeHost(inputHostPinned);
    cudaFreeHost(filterHostPinned);
    cudaFreeHost(outputHostPinned);
    cudaFreeHost(biasHostPinned);

    // Destroy stream
    cudaStreamDestroy(stream);
    
    cudaMemGetInfo(&freeMem, &totalMem);
    //cout << "Free Memory after deallocation: " << freeMem << " / " << totalMem << " bytes\n";
    return true;
}


bool calcInputGradHost(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultInputGrad, // same dim as input
    int padding,
    int stride) {
    
    size_t freeMem, totalMem; // Used for debugging memory leaks
    cudaMemGetInfo(&freeMem, &totalMem);
    //cout << "Free Memory before allocation: " << freeMem << " / " << totalMem << " bytes\n";
    size_t inputAllocSize = origFilterInNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    size_t filterAllocSize = origFilterOutNum * origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth * sizeof(float);
    size_t lossPrevGradAllocSize = origFilterOutNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    size_t resultInputGradAllocSize = inputAllocSize; // Same size because same-padding

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

    cudaError_t mallocError;
    mallocError = cudaMallocHost((void**)&inputHostPinned, inputAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&filterHostPinned, filterAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&lossPrevGradHostPinned, lossPrevGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&resultInputGradHostPinned, resultInputGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }

    mallocError = cudaMalloc(&inputDeviceData, inputAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&filterDeviceData, filterAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&lossPrevGradDeviceData, lossPrevGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&resultInputGradDeviceData, resultInputGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }

    memcpy(inputHostPinned, input, inputAllocSize);
    memcpy(filterHostPinned, rotatedFilters, filterAllocSize);
    memcpy(lossPrevGradHostPinned, lossPrevGrad, lossPrevGradAllocSize);

    cudaMemcpyAsync(inputDeviceData, inputHostPinned, inputAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(filterDeviceData, filterHostPinned, filterAllocSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(lossPrevGradDeviceData, lossPrevGradHostPinned, lossPrevGradAllocSize, cudaMemcpyHostToDevice, stream);
    
    calcInputGradKernel<<<gridDim, blockDim, 0, stream>>>(inputDeviceData, origFilterInNum, inputDepth, inputHeight, inputWidth,
        lossPrevGradDeviceData, origFilterOutNum, 
        filterDeviceData, origFilterDepth, origFilterHeight, origFilterWidth,
        resultInputGradDeviceData,
        padding,
        stride);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "Error: CUDA kernel failed: " << cudaGetErrorString(error) << "\n";
        return false;
    }
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
    
    cudaMemGetInfo(&freeMem, &totalMem);
    //cout << "Free Memory after deallocation: " << freeMem << " / " << totalMem << " bytes\n";
    return true;
}

bool calcFilterGradHost(float* input, int origFilterInNum, int inputDepth, int inputHeight, int inputWidth,
    float* lossPrevGrad, int origFilterOutNum, //same dim as input bc same padding
    float* rotatedFilters, int origFilterDepth, int origFilterHeight, int origFilterWidth,
    float* resultGrad, // same dim as input
    int padding,
    int stride) {
    
    size_t freeMem, totalMem; // Used for debugging memory leaks
    cudaMemGetInfo(&freeMem, &totalMem);
    //cout << "Free Memory before allocation: " << freeMem << " / " << totalMem << " bytes\n";
    
    size_t inputAllocSize = origFilterInNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    size_t filterAllocSize = origFilterOutNum * origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth * sizeof(float);
    size_t lossPrevGradAllocSize = origFilterOutNum * inputDepth * inputHeight * inputWidth * sizeof(float);
    size_t resultFilterGradAllocSize = filterAllocSize; // Same size because same-padding

    int totalThreads = origFilterHeight * origFilterWidth; // Total amount of threads per block
    int maxThreads = min(1024, totalThreads); // Actual amount of threads used per block
    int blockH = min(origFilterHeight, maxThreads); // Height of each block
    int blockW = max(1, maxThreads / blockH); // Width of each block. 1 if unnecessary

    dim3 gridDim(origFilterOutNum, origFilterInNum, origFilterDepth * ceil(float(totalThreads) / 1024));
    dim3 blockDim(blockH, blockW, 1);

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

    cudaError_t mallocError;
    mallocError = cudaMallocHost((void**)&inputHostPinned, inputAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&filterHostPinned, filterAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&lossPrevGradHostPinned, lossPrevGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMallocHost((void**)&resultFilterGradHostPinned, resultFilterGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    
    mallocError = cudaMalloc(&inputDeviceData, inputAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&filterDeviceData, filterAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&lossPrevGradDeviceData, lossPrevGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }
    mallocError = cudaMalloc(&resultFilterGradDeviceData, resultFilterGradAllocSize);
    if (mallocError != cudaSuccess) {
        cerr << "Error: CUDA memory allocation failure: " << cudaGetErrorString(mallocError) << "\n";
        return false;
    }

    
   
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
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "Error: CUDA kernel failed: " << cudaGetErrorString(error) << "\n";
        return false;
    }
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
    cudaMemGetInfo(&freeMem, &totalMem);
    //cout << "Free Memory after deallocation: " << freeMem << " / " << totalMem << " bytes\n";
    return true;
}


#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <iomanip>
#include "framework.h"
#include "ISLES.h"
#include "CNN.h"

using namespace std;
using namespace C;

vector<vector<vector<int>>> input;

// Convert 1D NifTI data to a 3D matrix
void CNN::convert1To3(vector<float>& voxels) {
    voxelsGrid = vector<vector<vector<float>>>(depth,
        vector<vector<float>>(height,
            vector<float>(width)));

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * (height * width) + y * width + x; // Calculate index in the 1D array
                voxelsGrid[z][y][x] = voxels[index]; 
            }
        }
    }
} 

// Read Header file
void CNN::readNiftiHeader(const string& filename, bool bFlair) { 
    bIsFlair = bFlair;
    ifstream file(filename, ios::binary); // Binary file
    if (!file.is_open()) {
        writeToLog(".nii File open Error");
    }

    NiftiHeader header;
    file.read(reinterpret_cast<char*>(&header), 348); // Read the first 348 bytes, which is the header
    if (file.fail()) {
        writeToLog("Error reading Header");
        file.close();
        return;
    }

    // Header metadata reading
    writeToLog("HEADER METADATA");
    writeToLog("Data type: " + to_string(header.datatype));
    writeToLog("Number of Dimensions: " + to_string(header.dim[0]) + "\nX: " + to_string(header.dim[1]) + "\nY: " + to_string(header.dim[2]) + "\nZ: " + to_string(header.dim[3]));
    writeToLog("pixX: " + to_string(header.pixdim[0]) + "\npixY: " + to_string(header.pixdim[1]) + "\npixZ: " + to_string(header.pixdim[2]));
    writeToLog("bit per voxel: " + to_string(header.bitpix));
    writeToLog("Voxel offset: " + to_string(header.vox_offset));
    writeToLog("scl_slope: " + to_string(header.scl_slope));
    writeToLog("scl_inter: " + to_string(header.scl_inter));
    writeToLog("Q Transform Code: " + to_string(header.qform_code));
    writeToLog("S Transform Code: " + to_string(header.sform_code));
    writeToLog("Affine Row X. 0: " + to_string(header.srow_x[0]) + "\nAffine Row X. 1: " + to_string(header.srow_x[1]) + "\nAffine Row X. 2: " + to_string(header.srow_x[2]) + "\nAffine Row X. 3: " + to_string(header.srow_x[3]));
    writeToLog("Affine Row Y. 0: " + to_string(header.srow_y[0]) + "\nAffine Row Y. 1: " + to_string(header.srow_y[1]) + "\nAffine Row Y. 2: " + to_string(header.srow_y[2]) + "\nAffine Row Y. 3: " + to_string(header.srow_y[3]));
    writeToLog("Affine Row Z. 0: " + to_string(header.srow_z[0]) + "\nAffine Row Z. 1: " + to_string(header.srow_z[1]) + "\nAffine Row Z. 2: " + to_string(header.srow_z[2]) + "\nAffine Row Z. 3: " + to_string(header.srow_z[3]));
    writeToLog("Intent Name: " + string(header.intent_name));
    writeToLog("Intent P1: " + to_string(header.intent_p1) + "\nIntent P2: " + to_string(header.intent_p2) + "\nIntent P3: " + to_string(header.intent_p3));
    writeToLog("/////////////////////////////////////////////////////////////////////////////////////////////////////////////////");
    endLine();
    
    //Reading bytes 349 onwards into a 3D matrix//
    
    width = header.dim[1]; // X
    height = header.dim[2]; // Y
    depth = header.dim[3]; // Z
    
    int numVoxels = width * height * depth; // Total number of voxels

    // Transformations and Resampling
    if (!bFlair) { // Is target file
        
        resultWidth = width; // If the file is not being resampled, set the target resample dimensions as own dimensions
        resultHeight = height; 
        resultDepth = depth;
        
        // AFFINE MATRIX
        writeToLog("Compiling ADC/DWI affine matrix.");
        vector<float> vecX = { header.srow_x[0], header.srow_x[1], header.srow_x[2], header.srow_x[3] };
        vector<float> vecY = { header.srow_y[0], header.srow_y[1], header.srow_y[2], header.srow_y[3] };
        vector<float> vecZ = { header.srow_z[0], header.srow_z[1], header.srow_z[2], header.srow_z[3] };
        targetAffineMatrix.push_back(vecX);
        targetAffineMatrix.push_back(vecY);
        targetAffineMatrix.push_back(vecZ);
        targetAffineMatrix.push_back(BottomAffineVector);
        writeToLog("Completed compiling ADC/DWI affine matrix.");
        writeToLog("Inverting ADC/DWI affine Matrix.");
        for (const auto& row : targetAffineMatrix) {
            for (const auto& element : row) {
                writeToLog(to_string(element));
            }
        }
        inverseAffine(targetAffineMatrix, targetAffineInverseMatrix); // Inverse Matrix
        writeToLog("Completed inverting matrix.");
    }
    else { // is flair file!!
        writeToLog("Resampling to Size (" + to_string(resultWidth) + ", " + to_string(resultHeight) + ", " + to_string(resultDepth) + ")."); // Resample

        //AFFINE MATRIX
        writeToLog("Compiling FLAIR affine matrix.");
        vector<float> vecX = { header.srow_x[0], header.srow_x[1], header.srow_x[2], header.srow_x[3] };
        vector<float> vecY = { header.srow_y[0], header.srow_y[1], header.srow_y[2], header.srow_y[3] };
        vector<float> vecZ = { header.srow_z[0], header.srow_z[1], header.srow_z[2], header.srow_z[3] };
        flairAffineMatrix.push_back(vecX);
        flairAffineMatrix.push_back(vecY);
        flairAffineMatrix.push_back(vecZ);
        flairAffineMatrix.push_back(BottomAffineVector);
        writeToLog("Completed compiling FLAIR affine matrix.");

        writeToLog("Multiplying Inverse of target affine matrix with FLAIR affine matrix.");
        matrixAffineMultiplication(targetAffineInverseMatrix, flairAffineMatrix, finalAffineMatrix);
        writeToLog("Multiplication complete.");
 
    }
    writeToLog("Total size: " + to_string(numVoxels));
    switch (header.datatype) {
    case 16:
        process16NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix); // If datatype = 16, which is float32, store as vector of floats directly
        break;
    case 64:
        process64NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix); // If datatype = 64, which is float64, store as vector of floats by converting doubles to floats
        break;
    default:
        process16NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix); // Defaults to 16
        return;
    }

}
// Process Nifti data (float)
void CNN::process16NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {

    vector<float> voxels(numVoxels); // Store data as float
    ifstream file(filename, ios::binary); 
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg); // Move to Offset
    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8); // Read the correct number of voxels
    writeToLog("Scaling.");
    for (auto& voxel : voxels) {
        voxel = voxel * scl_slope + scl_inter; // Scale data using Header values
    }
    writeToLog("Scaling finished.");

    writeToLog("Converting to 3D.");
    convert1To3(voxels); // Convert to 3D vector
    writeToLog("Convert finished. Ready for Normalisation.");
    if (bIsFlair) { 
        writeToLog("Normalising Transformed FLAIR grid."); 
        normalise(transformGrid); 
    } else {
        writeToLog("Normalising ADC/DWI grid."); normalise(voxelsGrid);
    }
    
    file.close();
}

// Process Nifti data (double)
void CNN::process64NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {

    vector<double> voxels(numVoxels);  // Temporarily store the data as double
    ifstream file(filename, ios::binary);
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg); // Move to Offset

    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8); //Read the correct number of voxels
    
    writeToLog("Scaling.");
    for (auto& voxel : voxels) {
        voxel = voxel * scl_slope + scl_inter;  // Scale data using Header values
    }
    writeToLog("Scaling finished.");

    writeToLog("Data is of type Double. Converting to float.");
    vector<float> floatVoxels(voxels.size());
    for (size_t i = 0; i < voxels.size(); ++i) {
        floatVoxels[i] = static_cast<float>(voxels[i]);  // Convert each voxel from double to float
    }
    writeToLog("Conversion to float complete.");

    writeToLog("Converting to 3D.");
    convert1To3(floatVoxels); // Convert to 3D vector
    writeToLog("Conversion to 3D complete. Ready for Normalisation.");

    if (bIsFlair) { 
        writeToLog("Applying affine matrix to grid.");
        applyAffineToGrid(voxelsGrid, transformGrid);
        writeToLog("Application complete.");

        writeToLog("Normalising Transformed FLAIR grid."); 
        normalise(transformGrid);        
    }
    else {
        writeToLog("Normalising ADC/DWI grid.");
        normalise(voxelsGrid);
    }
    file.close();
}

// ReLU activation function
float CNN::relu(float x) {
    return max(0.0f, x);
}

// Normalises all values into [0,1]
void CNN::normalise(vector<vector<vector<float>>>& grid) {
    float max_value = 0;
    for (const auto& slice : grid) {
        for (const auto& row : slice) {
            for (const auto& voxel : row) {
                max_value = max(max_value, voxel); // Gets maximum value
            }
        }
    }

    
    for (auto& slice : grid) {
        for (auto& row : slice) {
            for (auto& voxel : row) {
                voxel /= max_value;  // Normalises to [0, 1] using maximum value
            }
        }
    }
    writeToLog("Normalisation Finished. Inserting to gridChannels as channel.");
    insertGrid(grid); // Inserts the current 3D grid into 4D tensor as a channel
    writeToLog("Insertion complete.");
}
 
// Convolution Layer. Applies 3D filters to all 3D input tensors into several 3D output tensors WIP
void CNN::convolve(
    const vector<vector<vector<vector<float>>>>& input, // 4D Input tensor
    const vector<vector<vector<vector<vector<float>>>>>& filters, // 4D Filters
    vector<vector<vector<vector<float>>>>& output, // 4D Output tensor
    int stride // Stride value
) {
    int inputChannels = input.size();
    int inputWidth = input[0].size();
    int inputHeight = input[0][0].size();
    int inputDepth = input[0][0][0].size();

    int filterWidth = filters[0].size();
    int filterHeight = filters[0][0].size();
    int filterDepth = filters[0][0][0].size();
    int outputChannels = filters[0][0][0][0].size();  // Number of filters needed

    int outputHeight = (inputHeight - filterHeight) / stride + 1;
    int outputWidth = (inputWidth - filterWidth) / stride + 1;
    int outputDepth = (inputDepth - filterDepth) / stride + 1;

    // Initialise  output tensor
    output.resize(outputHeight, vector<vector<vector<float>>>(
        outputWidth, vector<vector<float>>(
            outputDepth, vector<float>(outputChannels, 0)
        )));

    // Convolution
    for (int f = 0; f < outputChannels; ++f) {  // Iterate over all filters
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                for (int k = 0; k < outputDepth; ++k) {
                    float sum = 0;

                    // Iterate over the input channels
                    for (int c = 0; c < inputChannels; ++c) {  // FLAIR, DWI, ADC
                        for (int m = 0; m < filterHeight; ++m) {
                            for (int n = 0; n < filterWidth; ++n) {
                                for (int p = 0; p < filterDepth; ++p) {
                                    // Calculate input indices
                                    int x = i * stride + m;
                                    int y = j * stride + n;
                                    int z = k * stride + p;

                                    // Convolve: multiply and accumulate
                                    sum += input[x][y][z][c] * filters[c][m][n][p][f];
                                }
                            }
                        }
                    }
                    output[i][j][k][f] = relu(sum); // Apply activation(ReLU) and store the result
                }
            }
        }
    }
}

void CNN::initialiseFilter(vector<vector<vector<vector<float>>>>& filter,
    int filterChannels, int filterWidth, int filterHeight, int filterDepth) {
    writeToLog("Initialising Filters.");

    // Initialise filter tensor
    filter.resize(filterChannels,
        vector<vector<vector<float>>>(filterWidth,
            vector<vector<float>>(filterHeight,
                vector<float>(filterDepth))));
    
    // Randomly initialize filter values
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);  // Random values between -1 and 1

    for (int f = 0; f < filterChannels; ++f) {
        for (int w = 0; w < filterWidth; ++w) {
            for (int h = 0; h < filterHeight; ++h) {
                for (int d = 0; d < filterDepth; ++d) {
                    filter[f][w][h][d] = dis(gen);  // Assign random value
                }
            }
        }
    }
    writeToLog("Filter Initialisation Complete.");
}

void CNN::clear() {
    gridChannels.clear(); // Clear Channels for next set of files
}

void CNN::insertGrid(const vector<vector<vector<float>>>& grid) {
    gridChannels.push_back(grid); // Push 3D grid into 4D tensor
}

// Trilinear Interpolation...from scratch i dont wanna do this anym
float triLerp(const vector<vector<vector<float>>>& originalImage, float x, float y, float z) {
    int x0 = static_cast<int>(std::floor(x));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(std::floor(y));
    int y1 = y0 + 1;
    int z0 = static_cast<int>(std::floor(z));
    int z1 = z0 + 1;

    float xd = x - x0;
    float yd = y - y0;
    float zd = z - z0;

    // Ensure indices are within bounds
    x0 = max(0, min(x0, (int)originalImage.size() - 1));
    x1 = max(0, min(x1, (int)originalImage.size() - 1));
    y0 = max(0, min(y0, (int)originalImage[0].size() - 1));
    y1 = max(0, min(y1, (int)originalImage[0].size() - 1));
    z0 = max(0, min(z0, (int)originalImage[0][0].size() - 1));
    z1 = max(0, min(z1, (int)originalImage[0][0].size() - 1));

    // Trilinear interpolation
    float c00 = originalImage[x0][y0][z0] * (1 - xd) + originalImage[x1][y0][z0] * xd;
    float c01 = originalImage[x0][y0][z1] * (1 - xd) + originalImage[x1][y0][z1] * xd;
    float c10 = originalImage[x0][y1][z0] * (1 - xd) + originalImage[x1][y1][z0] * xd;
    float c11 = originalImage[x0][y1][z1] * (1 - xd) + originalImage[x1][y1][z1] * xd;

    float c0 = c00 * (1 - yd) + c10 * yd;
    float c1 = c01 * (1 - yd) + c11 * yd;

    return c0 * (1 - zd) + c1 * zd;
}

// TRANSFORMS //

//Get Inverse of Affine matrix (4x4)
bool CNN::inverseAffine(const vector<vector<float>>& mat, vector<vector<float>>& result) {
    float det = 0;
    // Get determinant
    for (int i = 0; i < 4; ++i) {
        det += mat[0][i] *
            (mat[1][(i + 1) % 4] * (mat[2][(i + 2) % 4] * mat[3][(i + 3) % 4] - mat[2][(i + 3) % 4] * mat[3][(i + 2) % 4]) -
                mat[1][(i + 2) % 4] * (mat[2][(i + 1) % 4] * mat[3][(i + 3) % 4] - mat[2][(i + 3) % 4] * mat[3][(i + 1) % 4]) +
                mat[1][(i + 3) % 4] * (mat[2][(i + 1) % 4] * mat[3][(i + 2) % 4] - mat[2][(i + 2) % 4] * mat[3][(i + 1) % 4]));
    }
    if (det == 0) {
        writeToLog("Affine matrix is not invertible.");
        return false;
    }
    // Algorithm I found to directly calculate the inverse matrix of 4x4
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] =
                ((mat[(j + 1) % 4][(i + 1) % 4] * (mat[(j + 2) % 4][(i + 2) % 4] * mat[(j + 3) % 4][(i + 3) % 4] - mat[(j + 2) % 4][(i + 3) % 4] * mat[(j + 3) % 4][(i + 2) % 4])) -
                (mat[(j + 1) % 4][(i + 2) % 4] * (mat[(j + 2) % 4][(i + 1) % 4] * mat[(j + 3) % 4][(i + 3) % 4] - mat[(j + 2) % 4][(i + 3) % 4] * mat[(j + 3) % 4][(i + 1) % 4])) +
                (mat[(j + 1) % 4][(i + 3) % 4] * (mat[(j + 2) % 4][(i + 1) % 4] * mat[(j + 3) % 4][(i + 2) % 4] - mat[(j + 2) % 4][(i + 2) % 4] * mat[(j + 3) % 4][(i + 1) % 4]))) *
                1.0f/det;
        }
    }
}

// Affine Matrix Multiply (4x4)
void CNN::matrixAffineMultiplication(const AffineMatrix mat1, const AffineMatrix mat2, AffineMatrix resultMat) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int u = 0; u < 4; u++) {
                resultMat[i][j] += mat1[i][u] * mat2[u][j];
            }
        }
    }
}

// Apply the final affine matrix to a point in a grid (i.e. flair file data)
vector<float> CNN::applyAffineToPoint(const AffineMatrix mat, float x, float y, float z) {
    vector<float> point = { x, y, z, 1 };
    vector<float> result(4, 0.0f);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i] += mat[i][j] * point[j];
        }
    }
    return { result[0], result[1], result[2] };
}

void CNN::applyAffineToGrid(const vector<vector<vector<float>>>& grid, vector<vector<vector<float>>>& result) {
    

    result.resize(resultWidth, vector<vector<float>>(resultHeight, vector<float>(resultDepth, 0.f)));
    for (int i = 0; i < resultWidth; ++i) {
        for (int j = 0; j < resultHeight; ++j) {
            for (int k = 0; k < resultDepth; ++k) {
                vector<float> resultPoint = applyAffineToPoint(finalAffineMatrix, i, j, k);
                float x = resultPoint[0];
                float y = resultPoint[1];
                float z = resultPoint[2];
                result[i][j][k] = triLerp(grid, x, y, z);
            }
        }
    }    
}


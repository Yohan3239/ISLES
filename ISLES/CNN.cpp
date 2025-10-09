#include <algorithm>
#include <limits> 
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <ppl.h>
#include <iomanip>
#include <omp.h>
#include "framework.h"
#include "cudaFunctions.h"
#include "ISLES.h"
#include "CNN.h"

using namespace std;
using namespace C;

// ALWAYS REMEMBER DATA IS IN ZYX NOT XYZ !!!!

// Convert 1D NifTI data to a 3D matrix
void CNN::convert1To3(vector<float>& voxels) {
    voxelsGrid = vector<vector<vector<float>>>(depth, vector<vector<float>>(height, vector<float>(width, 0.f)));

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                // 1D -> 3D
                int index = z * (height * width) + y * width + x;
                voxelsGrid[z][y][x] = voxels[index];
            }
        }
    }
}

bool CNN::readNifti(const string& filename, bool bFlair) { 
    ifstream file(filename, ios::binary); // Read in binary as file is in binary
    if (!file.is_open()) {
        cerr << "Error: .nii open failure\n"; // Error handling
        return false;
    }

    NiftiHeader header;
    file.read(reinterpret_cast<char*>(&header), 348); // Read the first 348 bytes, which is the header
    if (file.fail()) {
        cerr << "Error: Could not read header\n"; // Error handling
        file.close();
        return false;
    }

    if (!bFlair && header.datatype == 64) { // Check if header is a ADC/DWI header
        resultHeader = header; 
    }

    width = header.dim[1]; // size of X
    height = header.dim[2]; // size of Y
    depth = header.dim[3]; // size of Z
    int numVoxels = width * height * depth; // Total number of voxels

    if (!bFlair) { // Is file with target voxel space

        vector<float> vecX = { header.srow_x[0], header.srow_x[1], header.srow_x[2], header.srow_x[3] };
        vector<float> vecY = { header.srow_y[0], header.srow_y[1], header.srow_y[2], header.srow_y[3] };
        vector<float> vecZ = { header.srow_z[0], header.srow_z[1], header.srow_z[2], header.srow_z[3] };
        // Initialise target affine matrix with data from header
        targetAffineMatrix.push_back(vecX);
        targetAffineMatrix.push_back(vecY);
        targetAffineMatrix.push_back(vecZ);
        targetAffineMatrix.push_back(BottomAffineVector); 

        if (!inverseAffine(targetAffineMatrix, targetAffineInverseMatrix)) return false; // Inverse Matrix

    }
    else { // is flair file!!
        // Get result dimensions from header of target file
        resultWidth = resultHeader.dim[1];
        resultHeight = resultHeader.dim[2];
        resultDepth = resultHeader.dim[3];
        
        vector<float> vecX = { header.srow_x[0], header.srow_x[1], header.srow_x[2], header.srow_x[3] };
        vector<float> vecY = { header.srow_y[0], header.srow_y[1], header.srow_y[2], header.srow_y[3] };
        vector<float> vecZ = { header.srow_z[0], header.srow_z[1], header.srow_z[2], header.srow_z[3] };
        // Initialise flair affine matrix with data from header

        flairAffineMatrix.push_back(vecX);
        flairAffineMatrix.push_back(vecY);
        flairAffineMatrix.push_back(vecZ);
        flairAffineMatrix.push_back(BottomAffineVector);

        // Using inverse of target and then original affine matrix, obtaining a matrix that maps original grid to target grid voxel space
        matrixAffineMultiplication(targetAffineInverseMatrix, flairAffineMatrix, finalAffineMatrix);
    }

    // Data reading from byte 349+
    switch (header.datatype) {
    case 16:
        // float, FLAIR
        if (!process16NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix)) return false; 
        break;
    case 64:
        // double -> float (data loss, but better this way because my laptop may not survive processing doubles, ADC/DWI
        if (!process64NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix)) return false; 
        break;
    case 512:
        // uint16 -> float, Ground truth mask
        if (!process512NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix)) return false;
        break;
    default:
        cerr << "Error: Unrecognised datatype.\n";
        return false;
    }
    return true;

}

// Process datatype 16 - float
bool CNN::process16NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {

    vector<float> voxels(numVoxels, -1.0f); // Store data as float, initialise as -1 to avoid 0 everywhere confusion
    ifstream file(filename, ios::binary); 
    if (!file.is_open()) {
        cerr << "Error: Failed to open file.\n"; // Error handling
        return false;
    }
    // Move to Offset (I think default 352)
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg); 
    if (file.tellg() != static_cast<streamoff>(vox_offset)) {
        cerr << "Error: File seek failed.\n"; // Error handling
        return false;
    }
    // Read the correct number of bits, number of voxels * bytes per voxel
    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8); 

    for (auto& voxel : voxels) {
        voxel = voxel * scl_slope + scl_inter; // Scale data using Header values (Seems to all be 1 and 0 though...)
    }

    convert1To3(voxels); // Converts to 3D vector, size [depth][height][width]

    if (!applyMatToGrid(voxelsGrid, transformGrid)) return false; // Applying matrix to FLAIR grid

    if (!normalise(transformGrid)) return false; // Normalisation and insertion
    file.close();
    return true;
}

// Process datatype 64 - double
bool CNN::process64NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {

    vector<double> voxels(numVoxels);  // Temporarily store the data as double
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Failed to open file.\n"; 
        return false;
    }
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg); 
    if (file.tellg() != static_cast<streamoff>(vox_offset)) {
        cerr << "Error: File seek failed.\n"; 
        return false;
    }
    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8);

    for (auto& voxel : voxels) {
        voxel = voxel * scl_slope + scl_inter;  
    }

    vector<float> floatVoxels(voxels.size());
    for (size_t i = 0; i < voxels.size(); ++i) {
        floatVoxels[i] = static_cast<float>(voxels[i]);  // Convert each voxel from double to float
    }

    convert1To3(floatVoxels);

    if (!normalise(voxelsGrid)) return false;
    file.close();
    return true;
}

// Process datatype 512 - int
bool CNN::process512NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {
    
    vector<uint16_t> voxels(numVoxels);  // int, but is only 0 and 1s anyway
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Failed to open file.\n"; 
        return false;
    }
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg); 
    if (file.tellg() != static_cast<streamoff>(vox_offset)) {
        cerr << "Error: File seek failed.\n";
        return false;
    }
    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8);

    for (auto& voxel : voxels) {
        voxel = voxel * scl_slope + scl_inter;
    }

    vector<float> floatVoxels(voxels.size());
    for (size_t i = 0; i < voxels.size(); ++i) {
        floatVoxels[i] = static_cast<float>(voxels[i]);  // Convert each voxel from uINT to float
    }

    convert1To3(floatVoxels);

    groundTruthGrid.insert(groundTruthGrid.begin(), voxelsGrid.begin(), voxelsGrid.end()); //copy voxelsgrid into gt grid#
    
    file.close();
    return true;
}

// ReLU function
float CNN::relu(float x) {
    return (x > 0) ? x : 0.01f * x;
}

// Sigmoid function
float CNN::sigmoid(float x)
{
    x = max(min(x, 10), -10);
    return 1.f / (1.f + exp(-x));
}

//not used rn maybe later for test purposes.
float CNN::tanhActivation(float x) {
    return (exp(x) - exp(x)) / (exp(x) + exp(x));
}

bool CNN::normalise(vector<vector<vector<float>>>& grid) {
    float max_value = -99999999.f;
    for (auto& slice : grid) {
        for (auto& row : slice) {
            for (auto& voxel : row) {
                max_value = max(max_value, voxel); // Gets maximum value of grid
            }
        }
    }
    if (max_value <= 0.f) return false;
    
    for (auto& slice : grid) {
        for (auto& row : slice) {
            for (auto& voxel : row) {
                voxel /= max_value;  // Normalises to [0, 1] by dividing all by the maximum value
            }
        }
    }
    insertGrid(grid); // Inserts the current 3D grid into 4D tensor as a channel, order should be ADC, DWI, FLAIR
    return true;
}

bool CNN::convolve(
    const vector<vector<vector<vector<float>>>>& inputChannels, // 4D Input tensor ([inputChannels][z][y][x])
    const vector<vector<vector<vector<vector<float>>>>>& filterChannels, // outputChannels * inputChannels amount of randomised 3D filters [outputChannels][inputChannels][z][y][x]
    vector<vector<vector<vector<float>>>>& outputChannels, // 4D Output tensor [outputChannels][z][y][x]
    int stride, // Stride value
    int padding, // Padding
    vector<float> bias
) { 
    // Input dimensions
    int inputNum = inputChannels.size();
    int depth = inputChannels[0].size(); 
    int height = inputChannels[0][0].size();
    int width = inputChannels[0][0][0].size();

    // Output & Filter dimensions
    int outputNum = filterChannels.size();
    int filterDepth = filterChannels[0][0].size();
    int filterHeight = filterChannels[0][0][0].size();
    int filterWidth = filterChannels[0][0][0][0].size();

    // Use formula of [(W−K+2P)/S]+1 for output size, but should be equal to input as same-padding is always used
    int outputDepth = (depth - filterDepth + 2 * padding) / stride + 1;
    int outputHeight = (height - filterHeight + 2 * padding) / stride + 1;
    int outputWidth = (width - filterWidth + 2 * padding) / stride + 1;

    outputChannels.resize(outputNum, vector<vector<vector<float>>>(outputDepth, vector<vector<float>>(outputHeight, vector<float>(outputWidth, 0.0f))));
    // Flatten both input and filters
    vector<float> flattenedInputVector = flatten4D(inputChannels);
    vector<float> flattenedFiltersVector = flatten5D(filterChannels);
    
    // Convert all into pointers
    float* output1D = new float[outputNum * outputDepth * outputHeight * outputWidth]();
    float* flattenedInput = flattenedInputVector.data();
    float* flattenedFilters = flattenedFiltersVector.data();
    float* biasArray = bias.data();

    // Convolve through CUDA
    if (!convolveHost(flattenedInput, inputNum, depth, height, width, flattenedFilters, outputNum, filterDepth, filterHeight, filterWidth, output1D, outputDepth, outputHeight, outputWidth, biasArray, stride, padding)) return false;

    // Convert to 3D again
    for (int x = 0; x < outputWidth; ++x) {
        for (int y = 0; y < outputHeight; ++y) {
            for (int z = 0; z < outputDepth; ++z) {
                for (int c = 0; c < outputNum; ++c) {
                    int index = (c * (outputDepth * outputHeight * outputWidth) + z * (outputHeight * outputWidth) + y * outputWidth + x);
                    outputChannels[c][z][y][x] = output1D[index];
                }
            }
        }
    }
    // Free memory of array
    delete[] output1D;
    return true;
}

// Flatten 4D grid into 1D
vector<float> CNN::flatten4D(const vector<vector<vector<vector<float>>>>& Initial) {
    vector<float> flattened;
    for (auto& c : Initial) {
        for (auto& z : c) {
            for (auto& y : z) {
                for (auto& x : y) {
                    flattened.push_back(x);
                }
            }
        }
    }
    return flattened;
}
// Flatten 5D grid into 1D
vector<float> CNN::flatten5D(const vector<vector<vector<vector<vector<float>>>>>& Initial) {
    vector<float> flattened;
    for (auto& oc : Initial) {
        for (auto& ic : oc) {
            for (auto& z : ic) {
                for (auto& y : z) {
                    for (auto& x : y) {
                        flattened.push_back(x);
                    }
                }
            }
        }
    }
    return flattened;
}

// Initialise Filters
void CNN::initialiseFilters(vector<vector<vector<vector<vector<float>>>>>& filterChannels, int numOfOutput, int numOfInput, int filterWidth, int filterHeight, int filterDepth, bool isReLU) {

    // Initialise multiple filter channels, 3 for initial convolution, 8, 16 etc for further
    // filterChannels: [Output Channels][Input Channels][depth][height][width]
    filterChannels.resize(numOfOutput, vector<vector<vector<vector<float>>>>(numOfInput, vector<vector<vector<float>>>(filterDepth, vector<vector<float>>(filterHeight, vector<float>(filterWidth)))));

    // Randomly initialize filter values
    random_device randomDevice;
    mt19937 gen(randomDevice());
    normal_distribution<float> dis(0.f, 1.f);
    float stddev = 0.f;
    if (isReLU) {
        // He Initialisation
        stddev = sqrt(2.f / (numOfInput * filterWidth * filterHeight * filterDepth));
    }
    else {
        // Xavier Initialisation
        stddev = sqrt(1.f / (numOfInput * filterWidth * filterHeight * filterDepth + numOfOutput * filterWidth * filterHeight * filterDepth));
    }
    dis = normal_distribution<float>(0.f, stddev);
    // Fill each value with random values
    for (auto& i : filterChannels) {
        for (auto& j : i) {
            for (auto& k : j) {
                for (auto& l : k) {
                    for (auto& m : l) {
                        m = dis(gen); 
                    }
                }
            }
        }
    }
}

// Clear for second ADC/DWI file
void CNN::clear() {

    depth = 1;
    height = 1;
    width = 1;
    resultDepth = 1;
    resultHeight = 1;
    resultWidth = 1;
    targetAffineMatrix.clear(); 
    flairAffineMatrix.clear(); 

}

// Push 3D grid into gridChannels
void CNN::insertGrid(const vector<vector<vector<float>>>& grid) {
    gridChannels.push_back(grid); 
}

// Trilinear Interpolation with caching
float CNN::triLerp(const vector<vector<vector<float>>>& inputGrid, float x, float y, float z, TriLerpCache& tlCache ) {
    
    // Nearest rounded down integer
    tlCache.z0 = static_cast<int>(floor(z)); 
    tlCache.y0 = static_cast<int>(floor(y));
    tlCache.x0 = static_cast<int>(floor(x));
    
    // Nearest rounded up integer
    tlCache.z1 = tlCache.z0 + 1; 
    tlCache.y1 = tlCache.y0 + 1;
    tlCache.x1 = tlCache.x0 + 1;

    // difference between z and floor integer
    tlCache.zd = (z - tlCache.z0) / (tlCache.z1 - tlCache.z0); // Denominator should be 1
    tlCache.yd = (y - tlCache.y0) / (tlCache.y1 - tlCache.y0);
    tlCache.xd = (x - tlCache.x0) / (tlCache.x1 - tlCache.x0);

    // Clamp values into [0, available grid size] to prevent index errors
    tlCache.z0 = max(0, min(tlCache.z0, inputGrid.size() - 1));
    tlCache.z1 = max(0, min(tlCache.z1, inputGrid.size() - 1));
    tlCache.y0 = max(0, min(tlCache.y0, inputGrid[0].size() - 1));
    tlCache.y1 = max(0, min(tlCache.y1, inputGrid[0].size() - 1));
    tlCache.x0 = max(0, min(tlCache.x0, inputGrid[0][0].size() - 1));
    tlCache.x1 = max(0, min(tlCache.x1, inputGrid[0][0].size() - 1));

    // Get the 8 integer coordinates surrounding the fractional coordinate
    // Those are reversed from normal trilinear interpolation as coordinates are written [z][y][x] instead of [x][y][z]
    float c000 = inputGrid[tlCache.z0][tlCache.y0][tlCache.x0]; 
    float c001 = inputGrid[tlCache.z0][tlCache.y0][tlCache.x1]; 
    float c010 = inputGrid[tlCache.z0][tlCache.y1][tlCache.x0]; 
    float c011 = inputGrid[tlCache.z0][tlCache.y1][tlCache.x1]; 
    float c100 = inputGrid[tlCache.z1][tlCache.y0][tlCache.x0]; 
    float c101 = inputGrid[tlCache.z1][tlCache.y0][tlCache.x1]; 
    float c110 = inputGrid[tlCache.z1][tlCache.y1][tlCache.x0]; 
    float c111 = inputGrid[tlCache.z1][tlCache.y1][tlCache.x1]; 

    // Calculate weights
    tlCache.w000 = (1 - tlCache.xd) * (1 - tlCache.yd) * (1 - tlCache.zd); 
    tlCache.w100 = tlCache.xd * (1 - tlCache.yd) * (1 - tlCache.zd);
    tlCache.w010 = (1 - tlCache.xd) * tlCache.yd * (1 - tlCache.zd);
    tlCache.w110 = tlCache.xd * tlCache.yd * (1 - tlCache.zd);
    tlCache.w001 = (1 - tlCache.xd) * (1 - tlCache.yd) * tlCache.zd;
    tlCache.w101 = tlCache.xd * (1 - tlCache.yd) * tlCache.zd;
    tlCache.w011 = (1 - tlCache.xd) * tlCache.yd * tlCache.zd;
    tlCache.w111 = tlCache.xd * tlCache.yd * tlCache.zd;

    // weights * corners
    return tlCache.w000 * c000 + tlCache.w001 * c001 + tlCache.w010 * c010 + tlCache.w011 * c011 +
        tlCache.w100 * c100 + tlCache.w101 * c101 + tlCache.w110 * c110 + tlCache.w111* c111;
}

//Get Inverse of Affine matrix (4x4)
bool CNN::inverseAffine(const vector<vector<float>>& mat, vector<vector<float>>& result) {
    // found better method to inverse exploiting the fact that it is an affine matrix
    float a = mat[0][0]; // Can be represented like [A][b]
    float b = mat[0][1]; //                         [0001] where A is a 3x3 rotation matrix and b is the translation part 
    float c = mat[0][2];
    float d = mat[1][0];
    float e = mat[1][1];
    float f = mat[1][2];
    float g = mat[2][0];
    float h = mat[2][1];
    float i = mat[2][2];

    float ta = mat[0][3]; // Translations
    float tb = mat[1][3];
    float tc = mat[2][3];

    float detM = a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g; // determinant
    if (abs(detM) < 1e-6) {
        cerr << "\nError: Determinant less or equal to 0. Cannot compute inverse.\n"; // Error handling
        return false;
    }

    // Hardcoded inverse
    vector<vector<float>> invM = { {(e * i - f * h), (c * h - b * i), (b * f - c * e)}, {(f * g - d * i), (a * i - c * g), (c * d - a * f)}, {(d * h - e * g), (b * g - a * h), (a * e - b * d)} }; 
    for (auto& item : invM) {
        for (auto& sub : item) {
            sub /= detM;
        }
    }
    
    vector<vector<float>> transOrig = { {ta}, {tb}, {tc} };
    vector<vector<float>> transM = { {0}, {0}, {0} };
    // Get -inverseA*b
    for (int i = 0; i < 3; ++i) {        
        for (int j = 0; j < 3; ++j) {
            transM[i][0] += -invM[i][j] * transOrig[j][0];
        }
    }
    // Create new inverse affine matrix 
    result = { {invM[0][0], invM[0][1], invM[0][2], transM[0][0]}, {invM[1][0], invM[1][1], invM[1][2], transM[1][0]}, {invM[2][0], invM[2][1], invM[2][2], transM[2][0]}, {0,0,0,1}};
    return true;
}

// Affine Matrix Multiplication
void CNN::matrixAffineMultiplication(const AffineMatrix& mat1, const AffineMatrix& mat2, AffineMatrix& resultMat) {
    resultMat = createDefaultAffineMatrix();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int u = 0; u < 4; u++) {
                // First row * First Column, First row * Second Column, ...Second row * First column, etc.
                resultMat[i][j] += mat1[i][u] * mat2[u][j]; 
            }
        }
    }
}

// Apply the final affine matrix to a point(x,y,z)
vector<float> CNN::applyMatToPoint(const AffineMatrix& mat, float x, float y, float z) {
    vector<float> point = { x, y, z, 1 };
    vector<float> result = { 0, 0 ,0 ,0 };
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i] += mat[i][j] * point[j]; // Matrix * Point
        }
    }
    return result;
}

bool CNN::applyMatToGrid(vector<vector<vector<float>>>& inputGrid, vector<vector<vector<float>>>& resultGrid) {
    // DESTINATION DRIVEN, as is most functions in this project
    // This prevents gaps in resultGrid by ensuring literally every point in resultGrid has a corresponding point in inputGrid.
    
    // Initialise resultGrid
    resultGrid.resize(resultDepth, vector<vector<float>>(resultHeight, vector<float>(resultWidth, 0.f)));

    // Compute the inverse matrix of the final transform(inputGrid -> resultGrid) for (resultGrid -> inputGrid).
    AffineMatrix inverseMatrix = createDefaultAffineMatrix(); // Didn't initialise this in header to avoid naming confusion...
    if (!inverseAffine(finalAffineMatrix, inverseMatrix)) return false;

    // Iterate over every coordinate in resultGrid and find its corresponding coordinate in inputGrid
    for (int k = 0; k < resultDepth; ++k) {
        for (int j = 0; j < resultHeight; ++j) {
            for (int i = 0; i < resultWidth; ++i) {
                // Maps coordinate in resultGrid to coordinate in inputGrid
                vector<float> inputCoords = applyMatToPoint(inverseMatrix, i, j, k);

                float x = inputCoords[0];
                float y = inputCoords[1];
                float z = inputCoords[2];

                // Use Trilinear interpolatation at those coords as they are not integers.
                CNN::TriLerpCache dummyCache; // Dummy cache because I don't need to cache values but it needs a parameter
                dummyCache.w000 = 0.0f;
                dummyCache.w001 = 0.0f;
                dummyCache.w010 = 0.0f;
                dummyCache.w011 = 0.0f;
                dummyCache.w100 = 0.0f;
                dummyCache.w101 = 0.0f;
                dummyCache.w110 = 0.0f;
                dummyCache.w111 = 0.0f;

                dummyCache.x0 = 0;
                dummyCache.x1 = 0;
                dummyCache.xd = 0.0f;
                dummyCache.y0 = 0;
                dummyCache.y1 = 0;
                dummyCache.yd = 0.0f;
                dummyCache.z0 = 0;
                dummyCache.z1 = 0;
                dummyCache.zd = 0.0f;

                float interpolatedValue = triLerp(inputGrid, x, y, z, dummyCache);

                // Store resultant value in resultGrid
                resultGrid[k][j][i] = interpolatedValue;
            }
        }
    }
    return true;
}

// Apply sigmoid to all values. Now with caching
void CNN::activateSigmoidOverChannels(vector<vector<vector<vector<float>>>>& inputChannels) { //changed to cache bc faster in backprop

    int channels = inputChannels.size();
    int depth = inputChannels[0].size();
    int height = inputChannels[0][0].size();
    int width = inputChannels[0][0][0].size();

    sigmoidCaches.resize(depth, vector<vector<float>>(height, vector<float>(width, 0.f)));

    #pragma omp parallel for collapse(4)
    for (int ic = 0; ic < channels; ++ic) {
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Cache results for use later on in backpropagation
                     sigmoidCaches[z][y][x] = sigmoid(inputChannels[ic][z][y][x]);
                     inputChannels[ic][z][y][x] = sigmoidCaches[z][y][x];
                }
            }
        }
    }
}

// Apply ReLU to all values
void CNN::activateReLUOverChannels(vector<vector<vector<vector<float>>>>& inputChannels) {
    for (auto& grid : inputChannels) {
        for (auto& slice : grid) {
            for (auto& line : slice) {
                for (auto& point : line) {
                    point = relu(point);
                }
            }
        }
    }
}

// Pooling Layer
void CNN::pool(
    const vector<vector<vector<vector<float>>>>& inputChannels, 
    vector<vector<vector<vector<float>>>>& outputChannels, 
    int poolWidth, int poolHeight, int poolDepth,
    int stride ) {
    
    int inputChannelNum = inputChannels.size();
    int inputDepth = inputChannels[0].size();
    int inputHeight = inputChannels[0][0].size();
    int inputWidth = inputChannels[0][0][0].size();

    // Output dim using valid padding
    int outputWidth = (inputWidth - poolWidth) / stride + 1;
    int outputHeight = (inputHeight - poolHeight) / stride + 1;
    int outputDepth = (inputDepth - poolDepth) / stride + 1;


    // Resize outputChannels
    outputChannels.resize(inputChannelNum, vector<vector<vector<float>>>(outputDepth, vector<vector<float>>(outputHeight, vector<float>(outputWidth, 0.0f))));
    #pragma omp parallel for collapse(4)
    for (int channel = 0; channel < inputChannelNum; ++channel) {
        for (int z = 0; z < outputDepth; ++z) {
            for (int y = 0; y < outputHeight; ++y) {
                for (int x = 0; x < outputWidth; ++x) {

                    float maxVal = -numeric_limits<float>::infinity();


                    for (int pz = 0; pz < poolDepth; ++pz) {
                        for (int py = 0; py < poolHeight; ++py) {
                            for (int px = 0; px < poolWidth; ++px) {
                                int inputZ = z * stride + pz; // times by stride + pooling
                                int inputY = y * stride + py;
                                int inputX = x * stride + px;
                                
                                if (inputZ < inputDepth && inputY < inputHeight && inputX < inputWidth) { // Clamping
                                    maxVal = max(maxVal, inputChannels[channel][inputZ][inputY][inputX]); 
                                }
                            }
                        }
                    }
                    outputChannels[channel][z][y][x] = maxVal;
                }
            }   
        }
    }
}

// Binary segmentation
void CNN::binarySegmentation(const vector<vector<vector<float>>>& inputGrid, vector<vector<vector<float>>>& outputGrid) {
    outputGrid.resize(inputGrid.size(), vector<vector<float>>(inputGrid[0].size(), vector<float>(inputGrid[0][0].size(), 0.0f)));
    for (int i = 0; i < inputGrid.size(); ++i) {
        for (int j = 0; j < inputGrid[0].size(); ++j) {
            for (int k = 0; k < inputGrid[0][0].size(); ++k) {
                // If less than 0.5, set as 0. If more than 0.5, set as 1.
                outputGrid[i][j][k] = (inputGrid[i][j][k] < 0.5f) ? 0.f : 1.f;
            }
        }
    }
}

// Upsampling layer
void CNN::upsample(const vector<vector<vector<float>>>& inputGrid, vector<vector<vector<float>>>& outputGrid) {
    int originalWidth = inputGrid[0][0].size();
    int originalHeight = inputGrid[0].size();
    int originalDepth = inputGrid.size();
    // create caches in the same dimensions of the result grid for backpropagation
    tlCaches.resize(resultDepth, vector<vector<TriLerpCache>>(resultHeight, vector<TriLerpCache>(resultWidth))); 
    outputGrid.resize(resultDepth, vector<vector<float>>(resultHeight, vector<float>(resultWidth, 0.f)));
    for (int k = 0; k < resultDepth; ++k) {
        for (int j = 0; j < resultHeight; ++j) {
            for (int i = 0; i < resultWidth; ++i) {
                float z = k * (originalDepth - 1.f) / (resultDepth - 1.f); // result point * ratio between dimensions
                float y = j * (originalHeight - 1.f) / (resultHeight - 1.f);
                float x = i * (originalWidth - 1.f) / (resultWidth - 1.f);
                // Use Trilinear interpolation at those coords as they are not integers
                float interpolatedValue = triLerp(inputGrid, x, y, z, tlCaches[k][j][i]); // also save to tlCaches

                // Store resultant value in resultGrid
                outputGrid[k][j][i] = interpolatedValue;
            }
        }
    }
}

// Composite Loss
float CNN::compLoss(const vector<vector<vector<float>>>& pGrid, const vector<vector<vector<float>>>& tGrid, float smooth, vector<vector<vector<float>>>& gGrid) {
    float ratio = 0.7f; // how much tversky : focal loss

    // tversky
    float falseNeg = 0.95f; // beta      
    float falsePos = 0.05f; // alpha

    float alpha = 0.99f; // focal loss
    float gamma = 4.f;


    int d = pGrid.size();
    int h = pGrid[0].size();
    int w = pGrid[0][0].size();
    vector<float> predGrid;
    vector<float> trueGrid;
    vector<float> gradGrid;
    for (int z = 0; z < d; ++z) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Flattening both into 1D data for more efficient accessing
                predGrid.push_back(pGrid[z][y][x]);
                trueGrid.push_back(tGrid[z][y][x]);

            }
        }
    }

    double intersection = 0;
    double denominator = 0;
    double D = 0;
    double sum = 0;
    int size = trueGrid.size();
    gradGrid.resize(size, 0.f);
    for (int i = 0; i < size; ++i) {
        float p = predGrid[i]; 
        float t = trueGrid[i];


        intersection += p * t;
        denominator += falsePos * p * (1 - t) + falseNeg * (1 - p) * t;

        float p_t = 0.f;
        p_t = (t == 1.f) ? p : 1 - p;

        sum += -alpha * pow(1 - p_t, gamma) * log(max(p_t, 1e-4f));  // Total focal loss


        // focal loss gradient
        float focalGrad = (t == 1.f) ? -alpha * (gamma * pow(1 - p_t, gamma-1) * (-1) * log(max(p_t, 1e-4f)) + (pow((1-p_t), gamma) / max(p_t, 1e-4f))) : alpha * (gamma * pow(1 - p_t, gamma - 1) * (-1) * log(max(p_t, 1e-4f)) + (pow((1 - p_t), gamma) / max(p_t, 1e-4f)));
        
        gradGrid[i] += (1 - ratio) * focalGrad;
    }
    denominator += smooth + intersection;
    // Tversky Loss
    float tverskyLoss = 1 - (intersection + smooth) / denominator;

    for (int i = 0; i < size; ++i) {
        float p = predGrid[i];
        float t = trueGrid[i];

        gradGrid[i] += ratio * -(t * denominator - (intersection + smooth) * (t * (1-falseNeg) + falsePos * (1-t))) / max(1e-6, pow(denominator, 2)); // tversky loss gradient

    }

    gGrid.resize(d, vector<vector<float>>(h, vector<float>(w, 0.f)));

    for (int z = 0; z < d; ++z) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // 1D -> 3D
                int index = z * (h * w) + y * w + x;

                // Clamp before insertion
                gGrid[z][y][x] = max(-20.f, min(gradGrid[index], 20.f));
            }
        }
    }
    float focalLoss = sum / size;
    cout << "\nTversky Loss: " << tverskyLoss << endl;
    cout << "\nFocal Loss: " << focalLoss << endl;

    float loss = ratio * tverskyLoss + (1 - ratio) * focalLoss;
    cout << "\nComposite Loss: " << loss << endl;
    cout << "\nIntersection: " << intersection << endl;
    return loss;

}

// Calculating gradients of loss wrt filters in convolution
bool CNN::calcFilterGradients(const vector<vector<vector<vector<float>>>>& input, 
    const vector<vector<vector<vector<float>>>>& lossPrevGrad, 
    const vector<vector<vector<vector<vector<float>>>>>& origFilters, 
    vector<vector<vector<vector<vector<float>>>>>& resultGrad, 
    int padding, int stride) {
    // Similar to convolve()
    int inputDepth = input[0].size();
    int inputHeight = input[0][0].size();
    int inputWidth = input[0][0][0].size();

    int origFilterOutNum = origFilters.size();
    int origFilterInNum = origFilters[0].size();
    int origFilterDepth = origFilters[0][0].size();
    int origFilterHeight = origFilters[0][0][0].size();
    int origFilterWidth = origFilters[0][0][0][0].size();

    resultGrad.resize(origFilterOutNum, vector<vector<vector<vector<float>>>>(origFilterInNum, vector<vector<vector<float>>>(origFilterDepth, vector<vector<float>>(origFilterHeight, vector<float>(origFilterWidth, 0.0f)))));
    
    
    vector<float> flattenedInputVector = flatten4D(input);
    vector<float> flattenedlossPrevGradVector = flatten4D(lossPrevGrad);
    vector<float> flattenedFiltersVector = flatten5D(origFilters);

    float* resultFilterGrad1D = new float[origFilterOutNum * origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth]();
    float* flattenedInput = flattenedInputVector.data();
    float* flattenedlossPrevGrad = flattenedlossPrevGradVector.data();
    float* flattenedFilters = flattenedFiltersVector.data();

    if (!calcFilterGradHost(flattenedInput, origFilterInNum, inputDepth, inputHeight, inputWidth, flattenedlossPrevGrad, origFilterOutNum, flattenedFilters, origFilterDepth, origFilterHeight, origFilterWidth, resultFilterGrad1D, padding, stride)) return false;
    
    for (int oc = 0; oc < origFilterOutNum; ++oc) {
        for (int ic = 0; ic < origFilterInNum; ++ic) {
            for (int z = 0; z < origFilterDepth; ++z) {
                for (int y = 0; y < origFilterHeight; ++y) {
                    for (int x = 0; x < origFilterWidth; ++x) {
                        int index = (oc * (origFilterInNum * origFilterDepth * origFilterHeight * origFilterWidth) 
                            + ic * (origFilterDepth * origFilterHeight * origFilterWidth) 
                            + z * (origFilterHeight * origFilterWidth) 
                            + y * origFilterWidth + x);

                        resultGrad[oc][ic][z][y][x] = resultFilterGrad1D[index];
                    }
                }
            }
        }
    }
    
    delete[] resultFilterGrad1D;
}

// GoL wrt convolution
bool CNN::calcInputGradients(const vector<vector<vector<vector<float>>>>& input, 
    const vector<vector<vector<vector<float>>>>& lossPrevGrad, 
    const vector<vector<vector<vector<vector<float>>>>>& origFilters, 
    vector<vector<vector<vector<float>>>>& resultInputGrad, 
    int padding, int stride) {
    // Similar to convolve() and calcFilterGradients()
    int origFilterHeight = origFilters[0][0][0].size();
    int origFilterWidth = origFilters[0][0][0][0].size();
    auto rotatedFilters = origFilters;
    
    // Rotate filters 180
    rotateFilter(rotatedFilters);

    int inputDepth = input[0].size();
    int inputHeight = input[0][0].size();
    int inputWidth = input[0][0][0].size();
    
    int origFilterOutNum = origFilters.size();
    int origFilterInNum = origFilters[0].size();
    int origFilterDepth = origFilters[0][0].size(); 

    resultInputGrad.resize(origFilterInNum, vector<vector<vector<float>>>(inputDepth, vector<vector<float>>(inputHeight, vector<float>(inputWidth, 0.0f))));
    
    vector<float> flattenedInputVector = flatten4D(input);
    vector<float> flattenedlossPrevGradVector = flatten4D(lossPrevGrad);
    vector<float> flattenedFiltersVector = flatten5D(rotatedFilters);

    float* resultInputGrad1D = new float[origFilterInNum * inputDepth * inputHeight * inputWidth]();
    float* flattenedInput = flattenedInputVector.data();
    float* flattenedlossPrevGrad = flattenedlossPrevGradVector.data();
    float* flattenedFilters = flattenedFiltersVector.data();

    if (!calcInputGradHost(flattenedInput, origFilterInNum, inputDepth, inputHeight, inputWidth, 
        flattenedlossPrevGrad, origFilterOutNum, 
        flattenedFilters, origFilterDepth, origFilterHeight, origFilterWidth, 
        resultInputGrad1D, padding, stride)) return false;
    
    for (int c = 0; c < origFilterInNum; ++c) {
        for (int z = 0; z < inputDepth; ++z) {
            for (int y = 0; y < inputHeight; ++y) {
                for (int x = 0; x < inputWidth; ++x) {
                    int index = (c * (inputDepth * inputHeight * inputWidth) + z * (inputHeight * inputWidth) + y * inputWidth + x);
                    resultInputGrad[c][z][y][x] = resultInputGrad1D[index];
                }
            }
        }
    }

    delete[] resultInputGrad1D;
}

void CNN::rotateFilter(vector<vector<vector<vector<vector<float>>>>>& filters) {
    // Reverse all vectors in all dimensions to rotate 180
    for (auto& out : filters) {
        for (auto& in : out) {
            for (auto& z : in) {
                for (auto& y : z) {
                    reverse(y.begin(), y.end());
                }
                reverse(z.begin(), z.end());
            }
            reverse(in.begin(), in.end());
        }
    }
}

void CNN::calcBiasGradients(const vector<vector<vector<vector<float>>>>& lossPrevGrad, vector<float>& resultBias) {
    
    int lossChannels = lossPrevGrad.size();
    int lossDepth = lossPrevGrad[0].size();
    int lossHeight = lossPrevGrad[0][0].size();
    int lossWidth = lossPrevGrad[0][0][0].size();

    resultBias.resize(lossChannels);
    // Sum over every gradient
    for (int oc = 0; oc < lossChannels; ++oc) {
        float sum = 0.f;
        for (auto& z : lossPrevGrad[oc]) {
            for (auto& y : z) {
                for (auto& x : y) {
                    sum += x;
                }
            }
        }
        resultBias[oc] = sum;
    }

}

// Gradients of Loss wrt upsampling layer
void CNN::backwardUpsample(const vector<vector<vector<float>>>& outputGrad, vector<vector<vector<TriLerpCache>>>& CacheGrid, vector<vector<vector<float>>>& inputGrad) {
    
    int outputDepth = outputGrad.size();
    int outputHeight = outputGrad[0].size();
    int outputWidth = outputGrad[0][0].size();

    int inputDepth = outputChannel[0].size();
    int inputHeight = outputChannel[0][0].size();
    int inputWidth = outputChannel[0][0][0].size();

    inputGrad.resize(inputDepth, vector<vector<float>>(inputHeight, vector<float>(inputWidth, 0.f)));
    for (int z = 0; z < outputDepth; ++z) {
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                
                const CNN::TriLerpCache& cache = CacheGrid[z][y][x];
                
                float gradOut = outputGrad[z][y][x];
               
                // distribute back to those 8 corners using cached output interpolation results from earlier upsampling
                inputGrad[cache.z0][cache.y0][cache.x0] += gradOut * cache.w000; // Add instead of redefine as multiple output coordinates can correspond to a single input coordinate
                inputGrad[cache.z0][cache.y0][cache.x1] += gradOut * cache.w001;
                inputGrad[cache.z0][cache.y1][cache.x0] += gradOut * cache.w010;
                inputGrad[cache.z0][cache.y1][cache.x1] += gradOut * cache.w011;
                inputGrad[cache.z1][cache.y0][cache.x0] += gradOut * cache.w100;
                inputGrad[cache.z1][cache.y0][cache.x1] += gradOut * cache.w101;
                inputGrad[cache.z1][cache.y1][cache.x0] += gradOut * cache.w110;
                inputGrad[cache.z1][cache.y1][cache.x1] += gradOut * cache.w111;
            }
        }
    }
    

}

// GoL wrt Sigmoid
void CNN::backwardSigmoid(const vector<vector<vector<float>>>& outputGrad, vector<vector<vector<float>>>& inputGrad) {

    int depth = outputGrad.size();
    int height = outputGrad[0].size();
    int width = outputGrad[0][0].size();

    inputGrad.resize(depth, vector<vector<float>>(height, vector<float>(width, 0.f)));
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Calculate gradient from cached data
                float temp = sigmoidCaches[z][y][x]; 
                inputGrad[z][y][x] = outputGrad[z][y][x] * (temp * (1 - temp)); 

            }
        }
    }
    
}
// GoL wrt ReLU
void CNN::backwardReLU(const vector<vector<vector<vector<float>>>>& outputGrad, const vector<vector<vector<vector<float>>>>& forwardInput, vector<vector<vector<vector<float>>>>& inputGrad) {
    int channels = outputGrad.size();
    int depth = outputGrad[0].size();
    int height = outputGrad[0][0].size();
    int width = outputGrad[0][0][0].size();

    inputGrad.resize(channels, vector<vector<vector<float>>>(depth, vector<vector<float>>(height, vector<float>(width, 0.0f))));
    for (int c = 0; c < channels; ++c) {
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Multiply outputGrad by derivative of leakyReLU
                    inputGrad[c][z][y][x] = (forwardInput[c][z][y][x] > 0) ? outputGrad[c][z][y][x] : 0.01f * outputGrad[c][z][y][x];
                }
            }
        }
    }

}

// Backpropagation through convolution
bool CNN::backwardConvolve(const vector<vector<vector<vector<float>>>>& outputGrad, 
    vector<vector<vector<vector<vector<float>>>>>& filters, 
    vector<vector<vector<vector<float>>>>& inputGrad, 
    vector<float>& bias, 
    const vector<vector<vector<vector<float>>>>& forwardInput, 
    const float learningRate, const int padding) {
    // Output dim
    int channels = outputGrad.size();
    int depth = outputGrad[0].size();
    int height = outputGrad[0][0].size();
    int width = outputGrad[0][0][0].size();
    
    // Filter dim
    int outputNum = filters.size();
    int inputNum = filters[0].size();
    int filterDepth = filters[0][0].size();
    int filterHeight = filters[0][0][0].size();
    int filterWidth = filters[0][0][0][0].size();
   
    vector<vector<vector<vector<vector<float>>>>> gradFilters;
    gradFilters.resize(outputNum, vector<vector<vector<vector<float>>>>(inputNum, vector<vector<vector<float>>> (filterDepth, vector<vector<float>>(filterHeight, vector<float>(filterWidth, 0.0f)))));
    
    vector<float> gradBias;
    gradBias.resize(bias.size(), 0.0f);

    inputGrad.resize(inputNum, vector<vector<vector<float>>>(depth, vector<vector<float>>(height, vector<float>(width, 0.f))));
    // calculate
    if (!calcInputGradients(forwardInput, outputGrad, filters, inputGrad, padding, 1)) return false; // UPDATE THIS IF CHANGING PADDING & STRIDE
    
    if (!calcFilterGradients(forwardInput, outputGrad, filters, gradFilters, padding, 1)) return false;
   
    #pragma omp parallel for collapse(5)
    for (int oc = 0; oc < outputNum; ++oc) {
        for (int ic = 0; ic < inputNum; ++ic) {
            for (int z = 0; z < filterDepth; ++z) {
                for (int y = 0; y < filterHeight; ++y) {
                    for (int x = 0; x < filterWidth; ++x) {
                        // Update filters using gradient
                        filters[oc][ic][z][y][x] -= learningRate * gradFilters[oc][ic][z][y][x]; 
                        
                    }
                }
            }
        }
    }
    
    calcBiasGradients(outputGrad, gradBias);
    // Use gradient to update biases
    for (int i = 0; i < bias.size(); ++i) {
        bias[i] -= learningRate * gradBias[i];
    }
    
}

// GoL wrt pooling layer
void CNN::backwardPool(const vector<vector<vector<vector<float>>>>& outputGrad, 
    const vector<vector<vector<vector<float>>>>& forwardInput, 
    const vector<vector<vector<vector<float>>>>& forwardOutput, 
    vector<vector<vector<vector<float>>>>& inputGrad, 
    int poolDepth, int poolHeight, int poolWidth, int stride) {
    int channels = forwardInput.size();
    int depth = forwardInput[0].size();
    int height = forwardInput[0][0].size();
    int width = forwardInput[0][0][0].size();

    inputGrad.resize(channels, vector<vector<vector<float>>>(depth, vector<vector<float>>(height, vector<float>(width, 0.0f))));
    for (int c = 0; c < channels; ++c) {
        for (int z = 0; z < forwardOutput[0].size(); ++z) {
            for (int y = 0; y < forwardOutput[0][0].size(); ++y) {
                for (int x = 0; x < forwardOutput[0][0][0].size(); ++x) {

                    int startZ = z * stride;
                    int startY = y * stride;
                    int startX = x * stride;
                    // Clamp inside grid
                    int endZ = min(startZ + poolDepth, depth);
                    int endY = min(startY + poolHeight, height);
                    int endX = min(startX + poolWidth, width);

                    float maxVal = forwardOutput[c][z][y][x];
                    for (int iz = startZ; iz < endZ; ++iz) {
                        for (int iy = startY; iy < endY; ++iy) {
                            for (int ix = startX; ix < endX; ++ix) {
                                if (fabs(forwardInput[c][iz][iy][ix] - maxVal) < 1e-6) { // Prevent logic errors from floating point errors
                                    inputGrad[c][iz][iy][ix] += outputGrad[c][z][y][x]; // Set gradient of that position
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Export to NifTI file
void const CNN::gridToNifti(const vector<vector<vector<float>>>& grid, const string& filename, const bool bin) {
    // Dimensions of grid
    int dimZ = grid.size();
    int dimY = grid[0].size();
    int dimX = grid[0][0].size();
    
    NiftiHeader header;


    header = resultHeader; // copy header from ADC/DWI header, bc same size and metadata except datatype and bitpix
     
    header.datatype = (bin) ? 2 : 16; // Binary and Float differ in header datatype
    header.bitpix = (bin) ? 8 : 32; // Same with bitpix

    int voxels = dimX * dimY * dimZ; // Number of voxels
    vector<float> flattenedData;
    
    for (int z = 0; z < dimZ; z++) {
        for (int y = 0; y < dimY; y++) {
            for (int x = 0; x < dimX; x++) {
                // flatten data to 1D and scale 0-1 to 0-255 from normalised data so viewers can have a high contrast
                flattenedData.push_back(grid[z][y][x]*255); 
            }
        }
    }

    ofstream out(filename, ios::binary);
    if (!out) {
        cerr << "Error: couldn't open file for writing nifti.\n";
        return;
    }

    out.write(reinterpret_cast<const char*>(&header), sizeof(NiftiHeader)); // Write header

    int padSize = static_cast<int>(header.vox_offset) - sizeof(NiftiHeader); // distance between header and offset (should be 8)
    if (padSize > 0) {  
        vector<char> pad(padSize, 0);
        out.write(pad.data(), padSize); // Catch up to offset
    }

    // Write voxel data in the right datatype
    if (bin) {
        vector<uint8_t> binData;
        for (float f : flattenedData) {
            binData.push_back(static_cast<uint8_t>(f));
        }
        out.write(reinterpret_cast<const char*>(binData.data()), flattenedData.size() * header.bitpix / 8);
    }
    else {
        out.write(reinterpret_cast<const char*>(flattenedData.data()), flattenedData.size() * header.bitpix / 8);
    }
    out.close();
    cout << "NIfTI file written successfully at " << filename << "\n";
}
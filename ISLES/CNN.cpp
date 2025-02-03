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
#include "framework.h"
#include "ISLES.h"
#include "CNN.h"


using namespace std;
using namespace C;

// ALWAYS REMEMBER DATA IS IN ZYX NOT XYZ

// Convert 1D NifTI data to a 3D matrix
void CNN::convert1To3(vector<float>& voxels) {
    // Resize into a 3D vector with the correct dimensions (depth, height, width)
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

////////////////////////
// Reading NIFTI file //
////////////////////////
void CNN::readNifti(const string& filename, bool bFlair) { 
    bIsFlair = bFlair; //is flair file?
    ifstream file(filename, ios::binary); // Binary file
    if (!file.is_open()) {
        writeToLog("Error: .nii open failure"); // For robustness
    }

    NiftiHeader header;
    file.read(reinterpret_cast<char*>(&header), 348); // Read the first 348 bytes, which is the header
    if (file.fail()) {
        writeToLog("Error: Could not read header"); // For robustness
        file.close();
        return;
    }

    // Header metadata reading into header struct (first 348 bytes)
    writeToLog("HEADER METADATA");
    writeToLog("Size of Header (check endianness...): " + to_string(header.sizeof_hdr));
    writeToLog("Data type: " + to_string(header.datatype));
    writeToLog("Number of Dimensions: " + to_string(header.dim[0]) + "\nX: " + to_string(header.dim[1]) + "\nY: " + to_string(header.dim[2]) + "\nZ: " + to_string(header.dim[3]));
    writeToLog("pixX: " + to_string(header.pixdim[1]) + "\npixY: " + to_string(header.pixdim[2]) + "\npixZ: " + to_string(header.pixdim[3]));
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
    width = header.dim[1]; // size of X
    height = header.dim[2]; // size of Y
    depth = header.dim[3]; // size of Z
    int numVoxels = width * height * depth; // Total number of voxels



    ////////////////////////////////////
    // Transformations and Resampling //
    ////////////////////////////////////
    if (!bFlair) { // Is file with target voxel space
        
        resultWidth = width; // If the file is not being resampled, set the target resample dimensions as own dimensions
        resultHeight = height; 
        resultDepth = depth;
        
        // AFFINE MATRIX
        writeToLog("Compiling ADC/DWI affine matrix.");

        vector<float> vecX = { header.srow_x[0], header.srow_x[1], header.srow_x[2], header.srow_x[3] };
        vector<float> vecY = { header.srow_y[0], header.srow_y[1], header.srow_y[2], header.srow_y[3] };
        vector<float> vecZ = { header.srow_z[0], header.srow_z[1], header.srow_z[2], header.srow_z[3] };
        // Initialise target affine matrix with data from header
        targetAffineMatrix.push_back(vecX);
        targetAffineMatrix.push_back(vecY);
        targetAffineMatrix.push_back(vecZ);
        targetAffineMatrix.push_back(BottomAffineVector); 

        writeToLog("Completed compiling ADC/DWI affine matrix.");
        inverseAffine(targetAffineMatrix, targetAffineInverseMatrix); // Inverse Matrix
        writeToLog("Completed inverting matrix.");
    }
    else { // is flair file!!
        writeToLog("Resampling to Size (" + to_string(resultWidth) + ", " + to_string(resultHeight) + ", " + to_string(resultDepth) + ").");

        //AFFINE MATRIX
        writeToLog("Compiling FLAIR affine matrix.");
        vector<float> vecX = { header.srow_x[0], header.srow_x[1], header.srow_x[2], header.srow_x[3] };
        vector<float> vecY = { header.srow_y[0], header.srow_y[1], header.srow_y[2], header.srow_y[3] };
        vector<float> vecZ = { header.srow_z[0], header.srow_z[1], header.srow_z[2], header.srow_z[3] };
        // Initialise target affine matrix with data from header
        flairAffineMatrix.push_back(vecX);
        flairAffineMatrix.push_back(vecY);
        flairAffineMatrix.push_back(vecZ);
        flairAffineMatrix.push_back(BottomAffineVector);
        writeToLog("Completed compiling FLAIR affine matrix.");

        writeToLog("Multiplying Inverse of target affine matrix with FLAIR affine matrix."); 

        // Using inverse of target and then original affine matrix, obtaining a matrix that maps original grid to target grid voxel space
        matrixAffineMultiplication(targetAffineInverseMatrix, flairAffineMatrix, finalAffineMatrix);
        writeToLog("Multiplication complete.");

    }

    /////////////////////////////////
    // Actual voxel reading (348+) //
    /////////////////////////////////
    switch (header.datatype) {
    case 16:
        // float
        process16NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix); 
        break;
    case 64:
        // double -> float (data loss, but better this wiay because my laptop may not survive processing doubles
        process64NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix); 
        break;
    case 512:
        // uint16 -> float
        process512NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix);
        break;
    default:
        // defaults to float
        process16NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix);
        return;
    }

}

////////////////////////////////
// Process Nifti data (float) //
////////////////////////////////
void CNN::process16NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {

    vector<float> voxels(numVoxels, -1.0f); // Store data as float, initialise as -1 to avoid 0 everywhere confusion
    ifstream file(filename, ios::binary); 
    if (!file.is_open()) {
        writeToLog("Error: Failed to open file."); // For robustness
        return;
    }
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg); // Move to Offset (I think default 352?)
    if (file.tellg() != static_cast<streamoff>(vox_offset)) {
        writeToLog("Error: File seek failed."); // For robustness
    }
    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8); // Read the correct number of bits, number of voxels * bits per voxel

    writeToLog("Scaling.");
    for (auto& voxel : voxels) {
        
        voxel = voxel * scl_slope + scl_inter; // Scale data using Header values (Seems to all be 1 and 0 though...)
        
    }
    writeToLog("Scaling finished.");
    
    
    writeToLog("Converting to 3D.");
    convert1To3(voxels); // Converts to 3D vector, size [depth][height][width]
    writeToLog("Convert finished. Ready for Normalisation.");
    if (bIsFlair) { // If flair, voxel space different to ADC/DWI, therefore must apply transform to map
        writeToLog("Applying affine matrix to grid.");
        applyMatToGrid(voxelsGrid, transformGrid); // Applying matrix to original grid
        writeToLog("Application complete.");

        writeToLog("Normalising Transformed FLAIR grid.");
        normalise(transformGrid); // Normalisation and insertion
    } else { // If not flair, just normalise and insert as channel.

        writeToLog("Normalising ADC/DWI grid."); 
        normalise(voxelsGrid); // Normalisation and insertion
    }
    file.close();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Process Nifti data (double), literally same as above but change values to float after reading //
///////////////////////////////////////////////////////////////////////////////////////////////////
void CNN::process64NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {

    vector<double> voxels(numVoxels);  // Temporarily store the data as double
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        writeToLog("Error: Failed to open file.");
        return;
    }
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg);
    if (file.tellg() != static_cast<streamoff>(vox_offset)) {
        writeToLog("Error: File seek failed.");
    }
    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8);

    writeToLog("Scaling.");
    for (auto& voxel : voxels) {
        voxel = voxel * scl_slope + scl_inter;  
    }
    writeToLog("Scaling finished.");

    writeToLog("Data is of type Double. Converting to float.");
    vector<float> floatVoxels(voxels.size());
    for (size_t i = 0; i < voxels.size(); ++i) {
        floatVoxels[i] = static_cast<float>(voxels[i]);  // Convert each voxel from double to float
    }
    writeToLog("Conversion to float complete.");

    writeToLog("Converting to 3D.");
    convert1To3(floatVoxels); // Generate grid using floats
    writeToLog("Conversion to 3D complete. Ready for Normalisation.");

    if (bIsFlair) { 
        writeToLog("Applying affine matrix to grid.");
        applyMatToGrid(voxelsGrid, transformGrid);
        writeToLog("Application complete.");

        writeToLog("Normalising Transformed FLAIR grid."); 
        normalise(transformGrid);        
    }
    else {
        writeToLog("Normalising ADC/DWI or GT mask grid.");
        normalise(voxelsGrid);
    }
    file.close();
}

////////////////////////////////////
// For the ground truth mask grid //
////////////////////////////////////
void CNN::process512NiftiData(const string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix) {

    vector<uint16_t> voxels(numVoxels);  // int, but is only 0 and 1s anyway
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        writeToLog("Error: Failed to open file.");
        return;
    }
    file.seekg(static_cast<streamoff>(vox_offset), ios::beg);
    if (file.tellg() != static_cast<streamoff>(vox_offset)) {
        writeToLog("Error: File seek failed.");
        return;
    }
    file.read(reinterpret_cast<char*>(voxels.data()), numVoxels * bitpix / 8);

    writeToLog("Scaling.");
    for (auto& voxel : voxels) {
        voxel = voxel * scl_slope + scl_inter;
    }
    writeToLog("Scaling finished.");

    writeToLog("Data is of type UINT. Converting to float.");
    vector<float> floatVoxels(voxels.size());
    for (size_t i = 0; i < voxels.size(); ++i) {
        floatVoxels[i] = static_cast<float>(voxels[i]);  // Convert each voxel from uINT to float
    }
    writeToLog("Conversion to float complete.");

    writeToLog("Converting to 3D.");
    convert1To3(floatVoxels); // Generate grid using floats
    writeToLog("Conversion to 3D complete.");

    groundTruthGrid.insert(groundTruthGrid.begin(), voxelsGrid.begin(), voxelsGrid.end()); //copy voxelsgrid into gt grid
    file.close();
}

// ReLU function
float CNN::relu(float x) {
    return max(0.f, x);
}

// Sigmoid function
float CNN::sigmoid(float x)
{
    return 1.f-(1.f / (1.f + exp(-x)));
}

//not used rn maybe later for test purposes.
float CNN::tanhActivation(float x) {
    return (exp(x) - exp(x)) / (exp(x) + exp(x));
}
///////////////////
// Normalisation //
///////////////////
void CNN::normalise(vector<vector<vector<float>>>& grid) {
    float max_value = -99999999.f;
    for (auto& slice : grid) {
        for (auto& row : slice) {
            for (auto& voxel : row) {
                max_value = max(max_value, voxel); // Gets maximum value of grid
            }
        }
    }
    writeToLog("Normalising Maximum value: " + to_string(max_value));
    
    for (auto& slice : grid) {
        for (auto& row : slice) {
            for (auto& voxel : row) {
                voxel /= max_value;  // Normalises to [0, 1] by dividing all by the maximum value
            }
        }
    }
    writeToLog("Normalisation Finished. Inserting to gridChannels as channel.");
    insertGrid(grid); // Inserts the current 3D grid into 4D tensor as a channel, order should be ADC, DWI, FLAIR, all mapped to the same voxel space.
    writeToLog("Insertion complete.");
}

///////////////////////
// Convolution Layer //
///////////////////////
void CNN::convolve(
    const vector<vector<vector<vector<float>>>>& inputChannels, // 4D Input tensor ([inputChannels][z][y][x])
    const vector<vector<vector<vector<vector<float>>>>>& filterChannels, // outputChannels * inputChannels amount of randomised 3D filters [outputChannels][inputChannels][z][y][x]
    vector<vector<vector<vector<float>>>>& outputChannels, // 4D Output tensor [outputChannels][z][y][x]
    int stride, // Stride value
    int padding // Padding so grid does not shrink
) {
    int inputNum = inputChannels.size();
    depth = inputChannels[0].size(); 
    height = inputChannels[0][0].size();
    width = inputChannels[0][0][0].size();

    

    // Number of input channels always 3!! nvm it isnt i forgor about mutiple layers
    int outputNum = filterChannels.size();
    int filterDepth = filterChannels[0][0].size();
    int filterHeight = filterChannels[0][0][0].size();
    int filterWidth = filterChannels[0][0][0][0].size();

    vector<float> bias; //bias 
    bias.resize(outputNum, 0.f);
    // Use formula of [(W−K+2P)/S]+1 for output size but im typically going to use same-padding so this is not rlly necessary
    int outputDepth = (depth - filterDepth + 2 * padding) / stride + 1;
    int outputHeight = (height - filterHeight + 2 * padding) / stride + 1;
    int outputWidth = (width - filterWidth + 2 * padding) / stride + 1;

    outputChannels.resize(outputNum, vector<vector<vector<float>>>(outputDepth, vector<vector<float>>(outputHeight, vector<float>(outputWidth, 0.0f))));
   

    for (int outChannel = 0; outChannel < outputNum; ++outChannel) {
        for (int z = 0; z < outputDepth; ++z) {
            for (int y = 0; y < outputHeight; ++y) {
                for (int x = 0; x < outputWidth; ++x) {
                    
                    float sum = 0.0f;

                    // Iterate over all input channels
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
                                        sum += inputChannels[inChannel][inputZ][inputY][inputX] * filterChannels[outChannel][inChannel][fz][fy][fx];
                                    }
                                }
                            }
                        }
                    }

                    // Store the result in the output tensor
                    outputChannels[outChannel][z][y][x] = sum + bias[outChannel];
                }
            }
        }
    }
}



////////////////////////
// Initialise Filters //
////////////////////////
void CNN::initialiseFilters(vector<vector<vector<vector<vector<float>>>>>& filterChannels, int numOfOutput, int numOfInput, int filterWidth, int filterHeight, int filterDepth, bool isReLU) {
    // Initialise multiple filter channels, 3 for initial convolution, 8, 16 etc for further? 
    // filterChannels: [Output Channels][Input Channels][depth][height][width]
    filterChannels.resize(numOfOutput, vector<vector<vector<vector<float>>>>(numOfInput, vector<vector<vector<float>>>(filterDepth, vector<vector<float>>(filterHeight, vector<float>(filterWidth)))));

    // Randomly initialize filter values
    random_device randomDevice;
    mt19937 gen(randomDevice());
    normal_distribution<float> dis(0.f, sqrt(2.f / (numOfInput * filterWidth * filterHeight * filterDepth)));
    if (isReLU) {
        // Produces random values between - 1 and 1...after testing found out uniform was horrible at giving any meaning results so using He/xavier intiialisation
        dis = normal_distribution<float>(0.f, sqrt(2.f / (numOfInput * filterWidth * filterHeight * filterDepth)));
    }
    else {
        dis = normal_distribution<float>(0.f, sqrt(2.f / ((numOfInput + numOfOutput) * filterWidth * filterHeight * filterDepth)));
    }

    for (auto& i : filterChannels) {
        for (auto& j : i) {
            for (auto& k : j) {
                for (auto& l : k) {
                    for (auto& m : l) {
                        m = dis(gen); // Fill each value
                    }
                }
            }
        }
    }
}


void CNN::clearAll() {
    gridChannels.clear(); // Clear entire 4D tensor for next set of files
}

void CNN::clear() {
    targetAffineMatrix.clear(); // Target(ADC/DWI) Affine Matrix
    flairAffineMatrix.clear(); // FLAIR Affine Matrix

    // Clear inverse for second ADC/DWI file to be the template since they have the same volume
    targetAffineInverseMatrix = createDefaultAffineMatrix();

    finalAffineMatrix = createDefaultAffineMatrix();

}

void CNN::insertGrid(const vector<vector<vector<float>>>& grid) {
    gridChannels.push_back(grid); // Push 3D grid into 4D tensor
}

/////////////////////////////
// Trilinear Interpolation //
/////////////////////////////
float CNN::triLerp(const vector<vector<vector<float>>>& inputGrid, float x, float y, float z) {
    
    
    int z0 = static_cast<int>(std::floor(z)); // Clamps to nearest rounded down integer
    int y0 = static_cast<int>(std::floor(y));
    int x0 = static_cast<int>(std::floor(x));
    
    int z1 = z0 + 1; // Clamps to nearest rounded up integer
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    float zd = (z - z0) / (z1 - z0); // differences
    float yd = (y - y0) / (y1 - y0);
    float xd = (x - x0) / (x1 - x0);

    // Clamp clamped values into [0, available grid size]
    z0 = max(0, min(z0, (int)inputGrid.size() - 1));
    z1 = max(0, min(z1, (int)inputGrid.size() - 1));
    y0 = max(0, min(y0, (int)inputGrid[0].size() - 1));
    y1 = max(0, min(y1, (int)inputGrid[0].size() - 1));
    x0 = max(0, min(x0, (int)inputGrid[0][0].size() - 1));
    x1 = max(0, min(x1, (int)inputGrid[0][0].size() - 1));

    // Get the 8 integer coordinates surrounding the fractional coordinate
    // Those are reversed from normal trilinear interpolation as coordinates are written [z][y][x] instead of [x][y][z]
    float c000 = inputGrid[z0][y0][x0]; // c000 on xyz
    float c001 = inputGrid[z0][y0][x1]; // c100
    float c010 = inputGrid[z0][y1][x0]; // c010 
    float c011 = inputGrid[z0][y1][x1]; // c110
    float c100 = inputGrid[z1][y0][x0]; // c001
    float c101 = inputGrid[z1][y0][x1]; // c101
    float c110 = inputGrid[z1][y1][x0]; // c011
    float c111 = inputGrid[z1][y1][x1]; // c111

    // Interpolate through X
    float c00 = c000 * (1.0f - xd) + c001 * xd; // c00
    float c10 = c100 * (1.0f - xd) + c101 * xd; // c01
    float c01 = c010 * (1.0f - xd) + c011 * xd; // c10
    float c11 = c110 * (1.0f - xd) + c111 * xd; // c11

    // Interpolate through Y
    float c0 = c00 * (1.0f - yd) + c01 * yd; // c0
    float c1 = c10 * (1.0f - yd) + c11 * yd; // c1

    // Interpolate through Z
    return c0 * (1 - zd) + c1 * zd;
}

////////////////
// TRANSFORMS //
////////////////

//Get Inverse of Affine matrix (4x4)
bool CNN::inverseAffine(const vector<vector<float>>& mat, vector<vector<float>>& result) {
    // found better method to inverse exploiting the fact that it is an affine matrix
    float a = mat[0][0]; // Can be represented like [M][t]
    float b = mat[0][1]; //                         [0001] where M is a 3x3 rotation matrix and t is the translation part 
    float c = mat[0][2];
    float d = mat[1][0];
    float e = mat[1][1];
    float f = mat[1][2];
    float g = mat[2][0];
    float h = mat[2][1];
    float i = mat[2][2];

    float ta = mat[0][3]; // 
    float tb = mat[1][3];
    float tc = mat[2][3];

    float detM = a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g; // determinant

    vector<vector<float>> invM = { {(e * i - f * h), (c * h - b * i), (b * f - c * e)}, {(f * g - d * i), (a * i - c * g), (c * d - a * f)}, {(d * h - e * g), (b * g - a * h), (a * e - b * d)} };
    for (auto& item : invM) {
        for (auto& sub : item) {
            sub /= detM;
        }
    }
    vector<vector<float>> transOrig = { {ta}, {tb}, {tc} };
    vector<vector<float>> transM = { {0}, {0}, {0} };
    for (int i = 0; i < 3; ++i) {        
        for (int j = 0; j < 3; ++j) {
            transM[i][0] += -invM[i][j] * transOrig[j][0];
        }
    }
    result = { {invM[0][0], invM[0][1], invM[0][2], transM[0][0]}, {invM[1][0], invM[1][1], invM[1][2], transM[1][0]}, {invM[2][0], invM[2][1], invM[2][2], transM[2][0]}, {0,0,0,1}};
    return true;
}

// Affine Matrix Multiplication (any type) // changed so i can use it to help invert the affine nvm it too stupid
void CNN::matrixAffineMultiplication(const AffineMatrix& mat1, const AffineMatrix& mat2, AffineMatrix& resultMat) {
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int u = 0; u < 4; u++) {
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

/////////////////////////////////////
// Apply transform to initial grid //
/////////////////////////////////////
void CNN::applyMatToGrid(vector<vector<vector<float>>>& inputGrid, vector<vector<vector<float>>>& resultGrid) {
    // DESTINATION DRIVEN //
    // This prevents gaps in resultGrid by ensuring literally every point in resultGrid has a corresponding point in inputGrid.
    
    // Initialise resultGrid
    resultGrid.resize(resultDepth, vector<vector<float>>(resultHeight, vector<float>(resultWidth, 0.f)));

    // Compute the inverse matrix of the final transform(inputGrid -> resultGrid) for (resultGrid -> inputGrid).
    AffineMatrix inverseMatrix = createDefaultAffineMatrix(); // Didn't initialise this in header to avoid naming confusion...
    inverseAffine(finalAffineMatrix, inverseMatrix);

    // Iterate over every coordinate in resultGrid and find its corresponding coordinate in inputGrid
    for (int k = 0; k < resultDepth; ++k) {
        for (int j = 0; j < resultHeight; ++j) {
            for (int i = 0; i < resultWidth; ++i) {
                // Maps point in resultGrid to point in inputGrid
                vector<float> inputCoords = applyMatToPoint(inverseMatrix, i, j, k);

                float x = inputCoords[0];
                float y = inputCoords[1];
                float z = inputCoords[2];

                // Use Trilinear interpolatation at those coords as they are not integers.
                float interpolatedValue = triLerp(inputGrid, x, y, z);

                // Store resultant value in resultGrid
                resultGrid[k][j][i] = interpolatedValue;
            }
        }
    }
}

void CNN::activateSigmoidOverChannels(vector<vector<vector<vector<float>>>>& inputChannels) {
    for (auto& grid : inputChannels) {
        for (auto& slice : grid) {
            for (auto& line : slice) {
                for (auto& point : line) {
                    point = sigmoid(point);
                }
            }
        }
    }
}

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

void CNN::pool(
    const vector<vector<vector<vector<float>>>>& inputChannels, 
    vector<vector<vector<vector<float>>>>& outputChannels,
    int poolWidth, // dim of the pooling window
    int poolHeight, 
    int poolDepth,  
    int stride 
) {
    int inputChannelNum = inputChannels.size();
    int inputDepth = inputChannels[0].size();
    int inputHeight = inputChannels[0][0].size();
    int inputWidth = inputChannels[0][0][0].size();

    // Output dim
    int outputWidth = (inputWidth - poolWidth) / stride + 1;
    int outputHeight = (inputHeight - poolHeight) / stride + 1;
    int outputDepth = (inputDepth - poolDepth) / stride + 1;

    // Resize outputChannels
    outputChannels.resize(inputChannelNum, vector<vector<vector<float>>>(outputDepth, vector<vector<float>>(outputHeight, vector<float>(outputWidth, 0.0f))));

    for (int channel = 0; channel < inputChannelNum; ++channel) {
        for (int z = 0; z < outputDepth; ++z) {
            for (int y = 0; y < outputHeight; ++y) {
                for (int x = 0; x < outputWidth; ++x) {

                    float maxVal = -99999999999999.f;

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

void CNN::binarySegmentation(const vector<vector<vector<float>>>& inputGrid, vector<vector<vector<float>>>& outputGrid) {
    outputGrid.resize(inputGrid.size(), vector<vector<float>>(inputGrid[0].size(), vector<float>(inputGrid[0][0].size(), 0.0f)));
    for (int i = 0; i < inputGrid.size(); ++i) {
        for (int j = 0; j < inputGrid[0].size(); ++j) {
            for (int k = 0; k < inputGrid[0][0].size(); ++k) {
                outputGrid[i][j][k] = (inputGrid[i][j][k] <= 0.5f) ? 0.f : 1.f;
            }
        }
    }
}

void CNN::upsample(const vector<vector<vector<float>>>& inputGrid, vector<vector<vector<float>>>& outputGrid) {
    int originalWidth = inputGrid[0][0].size();
    int originalHeight = inputGrid[0].size();
    int originalDepth = inputGrid.size();

    outputGrid.resize(resultDepth, vector<vector<float>>(resultHeight, vector<float>(resultWidth, 0.f)));
    for (int k = 0; k < resultDepth; ++k) {
        for (int j = 0; j < resultHeight; ++j) {
            for (int i = 0; i < resultWidth; ++i) {
                float z = k * (originalDepth - 1.f) / (resultDepth - 1.f); // result point times ratio
                float y = j * (originalHeight - 1.f) / (resultHeight - 1.f);
                float x = i * (originalWidth - 1.f) / (resultWidth - 1.f);
                // Use Trilinear interpolation at those coords as they are not integers.
                float interpolatedValue = triLerp(inputGrid, x, y, z);

                // Store resultant value in resultGrid
                outputGrid[k][j][i] = interpolatedValue;
            }
        }
    }
}

long double CNN::crossEntropyLoss(const vector<vector<vector<float>>>& inputGrid, const vector<vector<vector<float>>>& gtMaskGrid) {
    int depth = inputGrid.size();
    int height = inputGrid[0].size();
    int width = inputGrid[0][0].size();

    gradientOfLoss.resize(depth, vector<vector<float>>(height, vector<float>(width, 0.f)));

    double sum = 0.L;
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x){
                if (inputGrid[z][y][x] < 0) {
                    writeToLog("negative mate");
                }
                
                if (gtMaskGrid[z][y][x] == 1) {
                    sum += log(inputGrid[z][y][x]);

                }
                else {
                    sum += log(1.f-inputGrid[z][y][x]);
                }
                gradientOfLoss[z][y][x] = -gtMaskGrid[z][y][x] / inputGrid[z][y][x];
            }
        }
    }
    writeToLog("Average loss:");
    return -sum/(depth*height*width);
}

/////////////////////
// Backpropagation //
/////////////////////
void CNN::calcFilterGradients(const vector<vector<vector<vector<float>>>>& input, const std::vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& origFilters, std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& resultGrad, int stride)
{
    
    int inputDepth = input[0].size();
    int inputHeight = input[0][0].size();
    int inputWidth = input[0][0][0].size();

    int origFilterOutNum = origFilters.size();
    int origFilterInNum = origFilters[0].size();
    int origFilterDepth = origFilters[0][0].size();
    int origFilterHeight = origFilters[0][0][0].size();
    int origFilterWidth = origFilters[0][0][0][0].size();

    resultGrad.resize(origFilterOutNum, vector<vector<vector<vector<float>>>>(origFilterInNum, vector<vector<vector<float>>>(origFilterDepth, vector<vector<float>>(origFilterHeight, vector<float>(origFilterWidth, 0.0f)))));
    
    for (int oc = 0; oc < origFilterOutNum; ++oc) {
        for (int ic = 0; ic < origFilterInNum; ++ic) {
            for (int z = 0; z < origFilterDepth; ++z) {
                for (int y = 0; y < origFilterHeight; ++y) {
                    for (int x = 0; x < origFilterWidth; ++x) {
                        float sum = 0.f;
                        
                        for (int gz = 0; gz < inputDepth; ++gz) {
                            for (int gy = 0; gy < inputHeight; ++gy) {
                                for (int gx = 0; gx < inputWidth; ++gx) {
                                    int zCord = stride * gz + z;
                                    int yCord = stride * gy + y;
                                    int xCord = stride * gx + x;
                                    if (zCord >= 0 && zCord < inputDepth &&
                                        yCord >= 0 && yCord < inputHeight &&
                                        xCord >= 0 && xCord < inputWidth) {
                                        sum += lossPrevGrad[oc][gz][gy][gx] * input[ic][zCord][yCord][xCord];
                                    } // gonna assume same-padding bc otherwise overcomplicated
                                    
                                    
                                }
                            }
                        }
                        resultGrad[oc][ic][z][y][x] = sum;
                    }
                }
            }
        }
    }
}

void CNN::calcInputGradients(const vector<vector<vector<vector<float>>>>& input, const vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& origFilters, std::vector<std::vector<std::vector<std::vector<float>>>>& resultInput, int stride)
{
    
    int inputDepth = input[0].size();
    int inputHeight = input[0][0].size();
    int inputWidth = input[0][0][0].size();

    int origFilterOutNum = origFilters.size();
    int origFilterInNum = origFilters[0].size();
    int origFilterDepth = origFilters[0][0].size();
    int origFilterHeight = origFilters[0][0][0].size();
    int origFilterWidth = origFilters[0][0][0][0].size();
    auto rotatedFilters = origFilters;
    rotateFilter(rotatedFilters);
    resultInput.resize(origFilterInNum, vector<vector<vector<float>>>(inputDepth, vector<vector<float>>(inputHeight, vector<float>(inputWidth, 0.0f))));

    
    for (int ic = 0; ic < origFilterInNum; ++ic) { //NOT ERROR DO NOT FIX
        for (int z = 0; z < inputDepth; ++z) {
            for (int y = 0; y < inputHeight; ++y) {
                for (int x = 0; x < inputWidth; ++x) {
                    float sum = 0.f;
                    for (int oc = 0; oc < origFilterOutNum; ++oc) {
                        for (int gz = 0; gz < inputDepth; ++gz) { //again do NOT fix input dim * loss dim bc SAME PADDING
                            for (int gy = 0; gy < inputHeight; ++gy) {
                                for (int gx = 0; gx < inputWidth; ++gx) {
                                    int zCord = z - stride * gz;
                                    int yCord = y - stride * gy;
                                    int xCord = x - stride * gx;
                                    if (zCord >= 0 && zCord < origFilterDepth &&
                                        yCord >= 0 && yCord < origFilterHeight &&
                                        xCord >= 0 && xCord < origFilterWidth) {
                                        sum += lossPrevGrad[oc][gz][gy][gx] * rotatedFilters[oc][ic][zCord][yCord][xCord];
                                    } // gonna assume same-padding bc otherwise overcomplicated


                                }
                            }
                        }
                    }
                    resultInput[ic][z][y][x] = sum;
                }
            }
        }
    }
}

void CNN::calcInputGradients(const vector<vector<vector<vector<float>>>>& input, const vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& origFilters, std::vector<std::vector<std::vector<std::vector<float>>>>& resultInput, int stride)
{

    int inputDepth = input[0].size();
    int inputHeight = input[0][0].size();
    int inputWidth = input[0][0][0].size();

    int origFilterOutNum = origFilters.size();
    int origFilterInNum = origFilters[0].size();
    int origFilterDepth = origFilters[0][0].size();
    int origFilterHeight = origFilters[0][0][0].size();
    int origFilterWidth = origFilters[0][0][0][0].size();
    auto rotatedFilters = origFilters;
    rotateFilter(rotatedFilters);
    resultInput.resize(origFilterInNum, vector<vector<vector<float>>>(inputDepth, vector<vector<float>>(inputHeight, vector<float>(inputWidth, 0.0f))));


    for (int ic = 0; ic < origFilterInNum; ++ic) { //NOT ERROR DO NOT FIX (for future me refrence)
        for (int z = 0; z < inputDepth; ++z) {
            for (int y = 0; y < inputHeight; ++y) {
                for (int x = 0; x < inputWidth; ++x) {
                    float sum = 0.f;
                    for (int oc = 0; oc < origFilterOutNum; ++oc) {
                        for (int gz = 0; gz < inputDepth; ++gz) { //again do NOT fix input dim * loss dim bc SAME PADDING
                            for (int gy = 0; gy < inputHeight; ++gy) {
                                for (int gx = 0; gx < inputWidth; ++gx) {
                                    int zCord = z - stride * gz;
                                    int yCord = y - stride * gy;
                                    int xCord = x - stride * gx;
                                    if (zCord >= 0 && zCord < origFilterDepth &&
                                        yCord >= 0 && yCord < origFilterHeight &&
                                        xCord >= 0 && xCord < origFilterWidth) {
                                        sum += lossPrevGrad[oc][gz][gy][gx] * rotatedFilters[oc][ic][zCord][yCord][xCord];
                                    } // gonna assume same-padding bc otherwise overcomplicated


                                }
                            }
                        }
                    }
                    resultInput[ic][z][y][x] = sum;
                }
            }
        }
    }

}

void CNN::rotateFilter(vector<vector<vector<vector<vector<float>>>>>& filters) {
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

void CNN::calcBiasGradients(const vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, std::vector<float>& resultBias) {

    int lossChannels = lossPrevGrad.size();
    int lossDepth = lossPrevGrad[0].size();
    int lossHeight = lossPrevGrad[0][0].size();
    int lossWidth = lossPrevGrad[0][0][0].size();

    resultBias.resize(lossChannels);

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

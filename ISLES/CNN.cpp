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

    ifstream file(filename, ios::binary);
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
    //Header metadata reading
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
    writeToLog("Quaternion offset X: " + to_string(header.qoffset_x) + "\nQuaternion offset Y: " + to_string(header.qoffset_y) + "\nQuaternion offset Z: " + to_string(header.qoffset_z));
    writeToLog("Quaternion Coeff. b: " + to_string(header.quatern_b) + "\nQuaternion Coeff. c: " + to_string(header.quatern_c) + "\nQuaternion Coeff. d: " + to_string(header.quatern_d));
    writeToLog("Intent Name: " + string(header.intent_name));
    writeToLog("Intent P1: " + to_string(header.intent_p1) + "\nIntent P2: " + to_string(header.intent_p2) + "\nIntent P3: " + to_string(header.intent_p3));
    writeToLog("/////////////////////////////////////////////////////////////////////////////////////////////////////////////////");
    endLine();
    //Reading bytes 349 onwards into a 3D matrix
    
    width = header.dim[1]; // X
    height = header.dim[2]; // Y
    depth = header.dim[3]; // Z
    
    int numVoxels = width * height * depth; // Total number of voxels

    // RESAMPLING STUFF
    if (!bFlair) {
        
        resultWidth = width; // If the file is not being resampled, set the target resample dimensions as own dimensions
        resultHeight = height;
        resultDepth = depth;

        // QUATERNION TO MATRIX
        targetQuaternion.b = header.quatern_b;
        targetQuaternion.c = header.quatern_c;
        targetQuaternion.d = header.quatern_d;
        targetQuaternion.a = sqrt(1.0f - header.quatern_b * header.quatern_b - header.quatern_c * header.quatern_c - header.quatern_d * header.quatern_d);
        quaternionToMatrix(targetQuaternion, targetRotMatrix);
        inverse(targetRotMatrix, targetRotInverseMatrix);

        // AFFINE MATRIX

    }
    else { // is flair file!!
        writeToLog("Resample to Size (" + to_string(resultWidth) + ", " + to_string(resultHeight) + ", " + to_string(resultDepth) + ")."); // Resample
        // Initialise flair Quaternion to be made into rotation matrix
        flairQuaternion.b = header.quatern_b;
        flairQuaternion.c = header.quatern_c;
        flairQuaternion.d = header.quatern_d;
        flairQuaternion.a = sqrt(1.0f - header.quatern_b * header.quatern_b - header.quatern_c * header.quatern_c - header.quatern_d * header.quatern_d);
        quaternionToMatrix(flairQuaternion, flairRotMatrix); // Copy matrix-fied quaternion into flair rotation matrix
        matrixRotMultiplication(flairRotMatrix, targetRotInverseMatrix, finalRotMatrix); // Multiply flair rotation matrix with the inverse of the target rotation inverse matrix and store it
        
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
    normalise(); // Normalise
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

    normalise(); // Normalise
    file.close();
}

// ReLU activation function
float CNN::relu(float x) {
    return max(0.0f, x);
}

// Normalises all values into [0,1]
void CNN::normalise() {
    writeToLog("Normalising.");
    float max_value = 0;
    for (const auto& slice : voxelsGrid) {
        for (const auto& row : slice) {
            for (const auto& voxel : row) {
                max_value = max(max_value, voxel); // Gets maximum value
            }
        }
    }

    
    for (auto& slice : voxelsGrid) {
        for (auto& row : slice) {
            for (auto& voxel : row) {
                voxel /= max_value;  // Normalises to [0, 1] using maximum value
            }
        }
    }
    writeToLog("Normalisation Finished. Inserting to gridChannels as channel.");
    insertGrid(voxelsGrid); // Inserts the current 3D grid into 4D tensor as a channel
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

void CNN::resample(const vector<vector<vector<vector<float>>>>& grid, const int resultWidth, const int resultheight, const int resultDepth) {
    
    
    int startWidth = grid.size();
    int startHeight = grid[0].size();
    int startDepth = grid[0][0].size();

    float xScale = static_cast<float>(startWidth) / resultWidth;
    float yScale = static_cast<float>(startHeight) / resultHeight;
    float zScale = static_cast<float>(startDepth) / resultDepth;


    //voxelsGrid.resize(resultWidth,             ADD AT END OF THIS THING
      //      vector<vector<float>>(resultHeight,
        //        vector<float>(resultDepth)));

    vector<vector<vector<float>>> resampledImage(resultWidth, // Resampled image to store result in. will be copied into voxelsGrid??
        vector<vector<float>>(resultHeight,
            vector<float>(resultDepth)));



}

// Trilinear Interpolation...from scratch i dont wanna do this anym
float triLerp(const std::vector<std::vector<std::vector<float>>>& originalImage, float x, float y, float z) {
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
void CNN::quaternionToMatrix(const CNN::Quaternion& quaternion, CNN::RotationMatrix& matrix) {

    // Quaternion into matrix
    matrix[0][0] = 1 - 2 * (quaternion.c * quaternion.c + quaternion.d * quaternion.d);
    matrix[0][1] = 2 * (quaternion.b * quaternion.c - quaternion.d * quaternion.a);
    matrix[0][2] = 2 * (quaternion.b * quaternion.d + quaternion.c * quaternion.a);

    matrix[1][0] = 2 * (quaternion.b * quaternion.c + quaternion.d * quaternion.a);
    matrix[1][1] = 1 - 2 * (quaternion.b * quaternion.b + quaternion.d * quaternion.d);
    matrix[1][2] = 2 * (quaternion.c * quaternion.d - quaternion.b * quaternion.a);

    matrix[2][0] = 2 * (quaternion.b * quaternion.d - quaternion.c * quaternion.a);
    matrix[2][1] = 2 * (quaternion.c * quaternion.d + quaternion.b * quaternion.a);
    matrix[2][2] = 1 - 2 * (quaternion.b * quaternion.b + quaternion.c * quaternion.c);

}

// Get INVERSE MATRIX
bool CNN::inverse(vector<vector<float>>& mat, vector<vector<float>>& result) {
    
    // DETERMINANT //
    double det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
        mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
        mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);

    if (det == 0) {
        return false;
    }

    // Adjugate Matrix
    vector<vector<float>> adj(3, std::vector<float>(3, 0.0f));

    adj[0][0] = mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1];
    adj[0][1] = mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2];
    adj[0   ][2] = mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1];

    adj[1][0] = mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2];
    adj[1][1] = mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0];
    adj[1][2] = mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2];

    adj[2][0] = mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0];
    adj[2][1] = mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1];
    adj[2][2] = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];

    // Divide the adjugate matrix by the determinant to get the inverse
    double invDet = 1.0 / det;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = adj[i][j] * invDet;
        }
    }
    return true;
}

void CNN::matrixRotMultiplication(CNN::RotationMatrix mat1, CNN::RotationMatrix mat2, CNN::RotationMatrix resultMat) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            for (int u = 0; u < 3; u++)
                resultMat[i][j] += mat1[i][u] * mat2[u][j];
        }
}

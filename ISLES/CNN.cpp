#include <algorithm>
#include <limits> 
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

// ALWAYS REMEMBER DATA IS IN ZYX NOT XYZ

// Convert 1D NifTI data to a 3D matrix
void CNN::convert1To3(vector<float>& voxels) {
    // Resize into a 3D vector with the correct dimensions (depth, height, width)
    voxelsGrid.resize(depth, vector<vector<float>>(height, vector<float>(width, 0.f)));

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int z = 0; z < depth; ++z) {
                // 1D -> 3D
                voxelsGrid[z][y][x] = voxels[(z * (height * width)) + (y * width) + x];
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
        writeToLog("Inverting ADC/DWI affine Matrix: ");
        for (const auto& row : targetAffineMatrix) {
            for (const auto& element : row) {
                writeToLog(to_string(element));
            }
        }
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

        writeToLog("Multiplication complete. Result: ");

        // Simple output sequence
        std::string output;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                output += std::to_string(targetAffineInverseMatrix[i][j]);
                if (j < 3) output += " "; // Space between elements
            }
            if (i < 3) output += "\n"; // New line between rows
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                output += std::to_string(flairAffineMatrix[i][j]);
                if (j < 3) output += " "; // Space between elements
            }
            if (i < 3) output += "\n"; // New line between rows
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                output += std::to_string(finalAffineMatrix[i][j]);
                if (j < 3) output += " "; // Space between elements
            }
            if (i < 3) output += "\n"; // New line between rows
        }
        writeToLog(output);
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
        // double -> float (data loss, but better this way because my laptop may not survive processing doubles
        process64NiftiData(filename, numVoxels, header.vox_offset, header.scl_slope, header.scl_inter, header.bitpix); 
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
        writeToLog("Normalising ADC/DWI grid.");
        normalise(voxelsGrid);
    }
    file.close();
}

// ReLU function to clamp above 0
float CNN::relu(float x) {
    return max(0.0f, x);
}

///////////////////
// Normalisation //
///////////////////
void CNN::normalise(vector<vector<vector<float>>>& grid) {
    float max_value = 0;
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
    const vector<vector<vector<vector<float>>>>& gridChannels, // 4D Input tensor ([Channels][z][y][x])
    const vector<vector<vector<vector<vector<float>>>>>& filterChannels, // [output][3][z][y][x]
    vector<vector<vector<vector<float>>>>& outputChannels, // 4D Output tensor (3D volume x Output)
    int stride // Stride value
) {
    // TODO: Add convolution 
}

////////////////////////
// Initialise Filters //
////////////////////////
void CNN::initialiseFilters(int numOfOutput, int filterWidth, int filterHeight, int filterDepth) {
    // Initialise multiple filter channels (input channel always 3 bc ADC, DWI, FLAIR)
    // filterChannels: [Output Channels][3][depth][height][width]
    filterChannels.resize(numOfOutput, vector<vector<vector<vector<float>>>>(3, vector<vector<vector<float>>>(filterDepth, vector<vector<float>>(filterHeight, vector<float>(filterWidth)))));

    // Randomly initialize filter values
    random_device randomDevice;
    mt19937 gen(randomDevice());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);  // Produces random values between -1 and 1
   
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
    int z1 = z0 + 1;                          // Clamps to nearest rounded up integer
    int y0 = static_cast<int>(std::floor(y));
    int y1 = y0 + 1;
    int x0 = static_cast<int>(std::floor(x));
    int x1 = x0 + 1;

    float zd = z - z0; // Difference between actual value and clamped floor value
    float yd = y - y0;
    float xd = x - x0;

    // Clamp clamped values into [0, available grid size]
    z0 = max(0, min(z0, (int)inputGrid.size() - 1));
    z1 = max(0, min(z1, (int)inputGrid.size() - 1));
    y0 = max(0, min(y0, (int)inputGrid[0].size() - 1));
    y1 = max(0, min(y1, (int)inputGrid[0].size() - 1));
    x0 = max(0, min(x0, (int)inputGrid[0][0].size() - 1));
    x1 = max(0, min(x1, (int)inputGrid[0][0].size() - 1));

    // Get the 8 integer coordinates surrounding the fractional coordinate
    float c000 = inputGrid[z0][y0][x0];
    float c001 = inputGrid[z0][y0][x1];
    float c010 = inputGrid[z0][y1][x0];
    float c011 = inputGrid[z0][y1][x1];
    float c100 = inputGrid[z1][y0][x0];
    float c101 = inputGrid[z1][y0][x1];
    float c110 = inputGrid[z1][y1][x0];
    float c111 = inputGrid[z1][y1][x1];

    // Interpolate through X
    float c00 = c000 * (1.0f - xd) + c001 * xd; // Those are reversed from normal trilinear interpolation as coordinates are written [z][y][x] instead of [x][y][z]
    float c01 = c010 * (1.0f - xd) + c011 * xd;
    float c10 = c100 * (1.0f - xd) + c101 * xd;
    float c11 = c110 * (1.0f - xd) + c111 * xd;

    // Interpolate through Y
    float c0 = c00 * (1.0f - yd) + c01 * yd;
    float c1 = c10 * (1.0f - yd) + c11 * yd;

    // Interpolate through Z
    return c0 * (1 - zd) + c1 * zd;
}

////////////////
// TRANSFORMS //
////////////////

//Get Inverse of Affine matrix (4x4)
bool CNN::inverseAffine(const vector<vector<float>>& mat, vector<vector<float>>& result) {

    // Compute determinant
    double det = 0.0; 
    for (int i = 0; i < 4; ++i) {
        double minor =
            mat[1][(i + 1) % 4] * (mat[2][(i + 2) % 4] * mat[3][(i + 3) % 4] - mat[2][(i + 3) % 4] * mat[3][(i + 2) % 4]) -
            mat[1][(i + 2) % 4] * (mat[2][(i + 1) % 4] * mat[3][(i + 3) % 4] - mat[2][(i + 3) % 4] * mat[3][(i + 1) % 4]) +
            mat[1][(i + 3) % 4] * (mat[2][(i + 1) % 4] * mat[3][(i + 2) % 4] - mat[2][(i + 2) % 4] * mat[3][(i + 1) % 4]);
        
        det += mat[0][i] * (i % 2 == 0 ? 1 : -1) * minor;
    }

    if (det == 0) { 
        writeToLog("Matrix is not invertible.");
        return false;
    }

    // Compute the cofactor matrix
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double minor =
                mat[(i + 1) % 4][(j + 1) % 4] * (mat[(i + 2) % 4][(j + 2) % 4] * mat[(i + 3) % 4][(j + 3) % 4] - mat[(i + 2) % 4][(j + 3) % 4] * mat[(i + 3) % 4][(j + 2) % 4]) -
                mat[(i + 1) % 4][(j + 2) % 4] * (mat[(i + 2) % 4][(j + 1) % 4] * mat[(i + 3) % 4][(j + 3) % 4] - mat[(i + 2) % 4][(j + 3) % 4] * mat[(i + 3) % 4][(j + 1) % 4]) +
                mat[(i + 1) % 4][(j + 3) % 4] * (mat[(i + 2) % 4][(j + 1) % 4] * mat[(i + 3) % 4][(j + 2) % 4] - mat[(i + 2) % 4][(j + 2) % 4] * mat[(i + 3) % 4][(j + 1) % 4]);

            // Apply sign changes and determinant
            result[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * minor / det;
        }
    }
    return true;
}

// Affine Matrix Multiplication (4x4)
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
    vector<float> result(4, 0.0f);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i] += mat[i][j] * point[j]; // Matrix * Point
        }
    }
    return { result[0], result[1], result[2] };
}

/////////////////////////////////////
// Apply transform to initial grid //
/////////////////////////////////////
void CNN::applyMatToGrid(
    vector<vector<vector<float>>>& inputGrid,
    vector<vector<vector<float>>>& resultGrid
) {
    // DESTINATION DRIVEN //
    // This prevents gaps in resultGrid by ensuring literally every point in resultGrid has a corresponding point in inputGrid.
    
    // Initialise resultGrid
    resultGrid.resize(resultDepth, vector<vector<float>>(resultHeight, vector<float>(resultWidth, 0.0f)));

    // Compute the inverse matrix of the final transform(inputGrid -> resultGrid) for (resultGrid -> inputGrid).
    AffineMatrix inverseMatrix = createDefaultAffineMatrix(); // Didn't initialise this in header to avoid naming confusion...
    bool invertible = inverseAffine(finalAffineMatrix, inverseMatrix);
    if (!invertible) {
        writeToLog("applyAffineToGrid: finalAffineMatrix is not invertible!");
        return;
    }

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

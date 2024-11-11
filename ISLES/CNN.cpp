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
void CNN::convert1To3(std::vector<float>& voxels) {
    voxelsGrid = std::vector<std::vector<std::vector<float>>>(depth,
        std::vector<std::vector<float>>(height,
            std::vector<float>(width)));

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * (height * width) + y * width + x; // Calculate index in the 1D array
                voxelsGrid[z][y][x] = voxels[index]; 
            }
        }
    }
} 


void CNN::readNiftiHeader(const string& filename, bool bResample) { // Read Header file

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
    
    width = header.dim[1];
    height = header.dim[2];
    depth = header.dim[3];
    
    int numVoxels = width * height * depth; // Total number of voxels

    if (!bResample) {
        resultWidth = width; // If the file is not being resampled, set the target resample dimensions as own dimensions
        resultHeight = height;
        resultDepth = depth;
    }
    else {
        writeToLog("Resample to Size (" + to_string(resultWidth) + ", " + to_string(resultHeight) + ", " + to_string(resultDepth) + ")."); // Resample
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
// Prints initial input
void CNN::printVoxelsGrid() {

    for (int z = 0; z < depth; ++z) {
        writeToLog("Slice " + to_string(z + 1));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                writeToLogNoLine(to_string(voxelsGrid[z][y][x]) + "    ");  // Print each value with a tab space
            }
            endLine();  // Move to the next row
        }
        endLine();  // Move to the next layer
    }
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

// Convolution Layer. Applies 3D filters to the 3D input matrix
void CNN::convolve(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input, // 4D Input tensor
    const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filters, // 4D Filters
    std::vector<std::vector<std::vector<std::vector<float>>>>& output, // 4D Output tensor
    int stride // Stride value
) {
    int inputHeight = input.size();
    int inputWidth = input[0].size();
    int inputDepth = input[0][0].size();
    int inputChannels = input[0][0][0].size();

    int filterHeight = filters[0].size();
    int filterWidth = filters[0][0].size();
    int filterDepth = filters[0][0][0].size();
    int outputChannels = filters[0][0][0][0].size();  // Number of filters needed

    int outputHeight = (inputHeight - filterHeight) / stride + 1;
    int outputWidth = (inputWidth - filterWidth) / stride + 1;
    int outputDepth = (inputDepth - filterDepth) / stride + 1;

    // Initialise  output tensor
    output.resize(outputHeight, std::vector<std::vector<std::vector<float>>>(
        outputWidth, std::vector<std::vector<float>>(
            outputDepth, std::vector<float>(outputChannels, 0)
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

                    // Apply activation (ReLU) and store the result
                    output[i][j][k][f] = relu(sum);
                }
            }
        }
    }
}

void CNN::initialiseFilter(vector<vector<vector<vector<float>>>>& filter,
    int filter_channels, int filter_height, int filter_width, int filter_depth) {
    writeToLog("Initialising Filters.");

    filter.resize(filter_channels,
        vector<vector<vector<float>>>(filter_height,
            vector<vector<float>>(filter_width,
                std::vector<float>(filter_depth))));
    
    // Randomly initialize filter values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);  // Random values between -1 and 1

    for (int f = 0; f < filter_channels; ++f) {
        for (int h = 0; h < filter_height; ++h) {
            for (int w = 0; w < filter_width; ++w) {
                for (int d = 0; d < filter_depth; ++d) {
                    filter[f][h][w][d] = dis(gen);  // Assign random value
                }
            }
        }
    }
    writeToLog("Filter Initialisation Complete.");
}

void CNN::clear() {
    gridChannels.clear();
}

void CNN::insertGrid(const std::vector<std::vector<std::vector<float>>>& grid) {
    gridChannels.push_back(grid);
}
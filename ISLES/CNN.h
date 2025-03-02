#pragma once
#include <vector>
#include <variant>
#include <string>

namespace C
{
    
    class CNN {

    public:

        // Is this file FLAIR?
        bool bIsFlair;

        // Nifti Header struct to read header data into usable variables
        struct NiftiHeader {
            int sizeof_hdr;
            char data_type[10];
            char db_name[18];
            int extents;
            short session_error;
            char regular;
            char dim_info;
            short dim[8];
            float intent_p1, intent_p2, intent_p3;
            short intent_code;
            short datatype;
            short bitpix;
            short slice_start;
            float pixdim[8];
            float vox_offset;
            float scl_slope;
            float scl_inter;
            short slice_end;
            char slice_code;
            char xyzt_units;
            float cal_max, cal_min;
            float slice_duration;
            float toffset;
            int glmax, glmin;
            char descrip[80];
            char aux_file[24];
            short qform_code, sform_code;
            float quatern_b, quatern_c, quatern_d;
            float qoffset_x, qoffset_y, qoffset_z;
            float srow_x[4], srow_y[4], srow_z[4];
            char intent_name[16];
            char magic[4];
        };

        struct TriLerpCache {
            int x0, x1, y0, y1, z0, z1;
            float xd, yd, zd;

            float w000, w001, w010, w011, w100, w101, w110, w111;
        };
        std::vector<std::vector<std::vector<TriLerpCache>>> tlCaches;
        std::vector<std::vector<std::vector<float>>> sigmoidCaches;
        // trying out 3 different activation functions?
        float relu(float x); // ReLU
        float sigmoid(float x); // sigmoid
        float tanhActivation(float x); // tanh


        void readNifti(const std::string& filename, bool bFlair); // Read Header data

        void convert1To3(std::vector<float>& voxels); // 1D to 3D data

        void normalise(std::vector<std::vector<std::vector<float>>>& grid); // Normalise between [,0,1]
        void clearAll(); // Clear all values for next case
        void clear(); // Clear for next file

        // Convolutional Layer, wip
        void convolve( 
            const std::vector<std::vector<std::vector<std::vector<float>>>>& input, // Input tensor
            const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filterChannels, // Filters
            std::vector<std::vector<std::vector<std::vector<float>>>>& output, // Output tensor
            int stride,
            int padding,
            std::vector<float> bias
        );
        std::vector<float> flatten4D(const std::vector<std::vector<std::vector<std::vector<float>>>>& Initial);
        std::vector<float> flatten5D(const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& Initial);

        void calcFilterGradients(const std::vector<std::vector<std::vector<std::vector<float>>>>& input, const std::vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& origFilters, std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& lossFilterGrad, int padding, int stride);
        void calcBiasGradients(const std::vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, std::vector<float>& resultBias);

        void calcInputGradients(const std::vector<std::vector<std::vector<std::vector<float>>>>& input, const std::vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& origFilters, std::vector<std::vector<std::vector<std::vector<float>>>>& resultInputGrad, int padding, int stride);
        void rotateFilter(std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filters);
        // Initialise randomised filter
        void initialiseFilters(std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filterChannels, int numOfOutput, int numOfInput, int filterWidth, int filterHeight, int filterDepth, bool isReLU);

        // Insert grid into 4D tensor
        void insertGrid(const std::vector<std::vector<std::vector<float>>>& grid);

        // Trilinear interpolation to fix fractional coordinates after transform between voxel spaces
        float triLerp(const std::vector<std::vector<std::vector<float>>>& inputGrid, float x, float y, float z, TriLerpCache& tlCache);
        // Processing different datatypes of NIFTI files
        void process16NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
        void process64NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
        void process512NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
    private:
        int width, height, depth = 1; // Current file dimensions
        int resultWidth, resultHeight, resultDepth = 1; // What the dimensions should be resampled to. Updated from the resultant file.
        
    public:
        
        std::vector<std::vector<std::vector<float>>> voxelsGrid; // Input data, dimension FLAIR
        std::vector<std::vector<std::vector<float>>> transformGrid; // Input data, transformed AND resampled using affine matrix, dimension ADC/DWI .
        std::vector<std::vector<std::vector<std::vector<float>>>> gridChannels;

        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> filterChannels1; // All filters
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> filterChannels2; // All filters
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> filterChannels3; // All filters

        std::vector<float> bias1; // All biases
        std::vector<float> bias2;
        std::vector<float> bias3;
        std::vector<float> finalBias;

        std::vector<std::vector<std::vector<std::vector<float>>>> convolvedChannels1; // Output through Convolutional layer
        std::vector<std::vector<std::vector<std::vector<float>>>> pooledChannels;
        std::vector<std::vector<std::vector<std::vector<float>>>> convolvedChannels2; // Output through Convolutional layer
      
        std::vector<std::vector<std::vector<std::vector<float>>>> convolvedChannels3; // Output through Convolutional layer
        
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> outputFilterChannels;
        std::vector<std::vector<std::vector<std::vector<float>>>> outputChannel; // FINAL output after all convolution
        std::vector<std::vector<std::vector<float>>> finalUpsampledGrid;
        std::vector<std::vector<std::vector<float>>> finalBinaryGrid; 

        std::vector<std::vector<std::vector<float>>> groundTruthGrid;
        std::vector<std::vector<std::vector<float>>> gradientOfLoss;

        // Affine Transformation Matrix (4x4)
        typedef std::vector<std::vector<float>> AffineMatrix;

        AffineMatrix createDefaultAffineMatrix() {
            return AffineMatrix(4, std::vector<float>(4, 0.0f)); // Initialize a 4x4 rotation matrix with 0.0f in all elements
        }
        
        
        // Default bottom row of affine matrix (0, 0, 0, 1) constant !!!

        const std::vector<float> BottomAffineVector = { 0.f, 0.f, 0.f, 1.f };

        // Affine matrix storage
        AffineMatrix targetAffineMatrix; // Target(ADC/DWI) Affine Matrix
        AffineMatrix flairAffineMatrix; // FLAIR Affine Matrix
        
        // Inverse
        AffineMatrix targetAffineInverseMatrix = createDefaultAffineMatrix();

        AffineMatrix finalAffineMatrix = createDefaultAffineMatrix(); 


        // in summary ...
        
        /// FLAIR affine matrix - A1. Maps FLAIR to common scanner space.
        ///ADC/DWI affine matrix - A2. Maps ADC/DWI to common scanner space.
        
        /// To get both in same voxel space (i.e. ADC/DWI voxel space), must apply (A2^-1 * A1) to FLAIR grid.
        /// Apply (A2^-1 * A1)^-1 to the ADC/DWI grid coordinates to map to FLAIR, then reverse to obtain values in FLAIR to resample to ADC/DWI voxel space.
        

        

        // Inverses affine matrix (or well, any 4x4 matrix)
        bool inverseAffine(const std::vector<std::vector<float>>& mat, std::vector<std::vector<float>>& result);
        // Multiplies two affine matrices (or well, any 4x4 matrices)
        void matrixAffineMultiplication(const AffineMatrix& mat1, const AffineMatrix& mat2, AffineMatrix& resultMat);

        // Apply matrix to a point.
        std::vector<float> applyMatToPoint(const AffineMatrix& mat, float x, float y, float z);
        
        // Uses above function to apply to grid
        void applyMatToGrid(std::vector<std::vector<std::vector<float>>>& grid, std::vector<std::vector<std::vector<float>>>& result);
        long double crossEntropyLoss(const std::vector<std::vector<std::vector<float>>>& inputGrid, const std::vector<std::vector<std::vector<float>>>& gtMaskGrid);
        void activateSigmoidOverChannels(std::vector<std::vector<std::vector<std::vector<float>>>>& inputChannels);
        void activateReLUOverChannels(std::vector<std::vector<std::vector<std::vector<float>>>>& inputChannels);
        void pool(const std::vector<std::vector<std::vector<std::vector<float>>>>& inputChannels, std::vector<std::vector<std::vector<std::vector<float>>>>& outputChannels, int poolWidth, int poolHeight, int poolDepth, int stride);
        
        void binarySegmentation(const std::vector<std::vector<std::vector<float>>>& inputGrid, std::vector<std::vector<std::vector<float>>>& outputGrid);

        void upsample(const std::vector<std::vector<std::vector<float>>>& inputGrid, std::vector<std::vector<std::vector<float>>>& outputGrid);
        void backwardUpsample(const std::vector<std::vector<std::vector<float>>>& outputGrad, std::vector<std::vector<std::vector<TriLerpCache>>>& CacheGrid, std::vector<std::vector<std::vector<float>>>& inputGrad);
        void backwardSigmoid(const std::vector<std::vector<std::vector<float>>>& outputGrad, std::vector<std::vector<std::vector<float>>>& inputGrad);
        void backwardReLU(const std::vector<std::vector<std::vector<std::vector<float>>>>& outputGrad, const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardInput, std::vector<std::vector<std::vector<std::vector<float>>>>& inputGrad);
        void backwardConvolve(const std::vector<std::vector<std::vector<std::vector<float>>>>& outputGrad, std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filters, std::vector<std::vector<std::vector<std::vector<float>>>>& inputGrad, std::vector<float>& bias, const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardInput, const float learningRate);
        void backwardPool(const std::vector<std::vector<std::vector<std::vector<float>>>>& outputGrad, const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardInput, const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardOutput, std::vector<std::vector<std::vector<std::vector<float>>>>& inputGrad, int poolDepth, int poolHeight, int poolWidth, int stride);
        // backprop grids
        std::vector<std::vector<std::vector<float>>> upsampleInputGrad;
        std::vector<std::vector<std::vector<float>>> finalSigmoidInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> finalConvolveInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> finalPoolInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> thirdReLUInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> thirdConvolveInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> secondReLUInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> secondConvolveInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> firstReLUInputGrad;
        std::vector<std::vector<std::vector<std::vector<float>>>> firstConvolveInputGrad;

        // debugging...
        void gradSum(const std::vector<std::vector<std::vector<std::vector<float>>>> grad);


        
    };
}

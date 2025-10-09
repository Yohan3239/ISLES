#pragma once
#include <vector>
#include <variant>
#include <string>

namespace C
{
    
    class CNN {

    public:

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

        NiftiHeader resultHeader; // save for exporting during testing
        // Struct for cache for backward upsampling quickly
        struct TriLerpCache {
            int x0, x1, y0, y1, z0, z1;
            float xd, yd, zd;

            float w000, w001, w010, w011, w100, w101, w110, w111;
        };
        // 3-order tensor of trilinear intepolation caches
        std::vector<std::vector<std::vector<TriLerpCache>>> tlCaches;

        // 3-order tensor of caches used for sigmoid
        std::vector<std::vector<std::vector<float>>> sigmoidCaches;

        // activation functions
        float relu(float x); // ReLU
        float sigmoid(float x); // sigmoid
        float tanhActivation(float x); // tanh, unused

        // Read Header data
        bool readNifti(const std::string& filename, bool bFlair); 

        // Processing different datatypes of NIFTI files
        bool process16NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
        bool process64NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
        bool process512NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
        
        // 1D to 3D data
        void convert1To3(std::vector<float>& voxels);

        // Normalise between [0,1]
        bool normalise(std::vector<std::vector<std::vector<float>>>& grid); 

        void clear(); // Clear for next file

        // Convolutional Layer
        bool convolve(
            const std::vector<std::vector<std::vector<std::vector<float>>>>& input, // Input tensor
            const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filterChannels, // Filters
            std::vector<std::vector<std::vector<std::vector<float>>>>& output, // Output tensor
            int stride,
            int padding,
            std::vector<float> bias
        );
        std::vector<float> flatten4D(const std::vector<std::vector<std::vector<std::vector<float>>>>& Initial); // 4D->1D
        std::vector<float> flatten5D(const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& Initial); // 5D->1D

        // Calculate gradients of loss wrt filters and update them
        bool calcFilterGradients(const std::vector<std::vector<std::vector<std::vector<float>>>>& input, // Input from forward pass
            const std::vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, // gradients of loss wrt the "previous" layer in the backward pass, so following layer in forward pass
            const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& origFilters, // Filters from forward pass to be updated
            std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& lossFilterGrad, // Gradients of loss wrt filters
            int padding, 
            int stride);

        // Calculate gradients of loss wrt biases and update them
        void calcBiasGradients(const std::vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, std::vector<float>& resultBias);

        // Calculate gradients of loss wrt the input of convolutions
        bool calcInputGradients(const std::vector<std::vector<std::vector<std::vector<float>>>>& input, const std::vector<std::vector<std::vector<std::vector<float>>>>& lossPrevGrad, const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& origFilters, std::vector<std::vector<std::vector<std::vector<float>>>>& resultInputGrad, int padding, int stride);
        
        // Rotate Filter by 180 degrees
        void rotateFilter(std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filters);

        // Initialise randomised filter
        void initialiseFilters(std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filterChannels, int numOfOutput, int numOfInput, int filterWidth, int filterHeight, int filterDepth, bool isReLU);

        // Insert grid into voxelGrids
        void insertGrid(const std::vector<std::vector<std::vector<float>>>& grid);

        // Trilinear interpolation to fix fractional coordinates after transform between voxel spaces
        float triLerp(const std::vector<std::vector<std::vector<float>>>& inputGrid, float x, float y, float z, TriLerpCache& tlCache);
        
    private:
        int width, height, depth = 1; // Current file dimensions
        int resultWidth, resultHeight, resultDepth = 1; // What the dimensions should be resampled to. Updated from the resultant file.
        
    public:
        
        std::vector<std::vector<std::vector<float>>> voxelsGrid; // Input data
        std::vector<std::vector<std::vector<float>>> transformGrid; // Input data, transformed and resampled using affine matrix
        std::vector<std::vector<std::vector<std::vector<float>>>> gridChannels; // All 3 Initial grids after input preprocessing

        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> filterChannels1; // filters
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> filterChannels2; 
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> filterChannels3; 
        std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> outputFilterChannels; // Final filters

        std::vector<float> bias1; // biases
        std::vector<float> bias2;
        std::vector<float> bias3;
        std::vector<float> finalBias;

        std::vector<std::vector<std::vector<std::vector<float>>>> convolvedChannels1; // Output of Convolutional layer 1
        
        std::vector<std::vector<std::vector<std::vector<float>>>> convolvedChannels2; // Output of Convolutional layer 2
      
        std::vector<std::vector<std::vector<std::vector<float>>>> convolvedChannels3; // Output of Convolutional layer 3

        std::vector<std::vector<std::vector<std::vector<float>>>> pooledChannels; // Output of pooling

       
        std::vector<std::vector<std::vector<std::vector<float>>>> outputChannel; // output of final convolution
        std::vector<std::vector<std::vector<float>>> finalUpsampledGrid; // Output of upsampling
        std::vector<std::vector<std::vector<float>>> finalBinaryGrid; // Output of binary segmentation

        std::vector<std::vector<std::vector<float>>> groundTruthGrid; // Ground truth mask
        std::vector<std::vector<std::vector<float>>> gradientOfLoss; // gradients of Loss wrt predicted values

        // Affine Transformation Matrix (4x4), honestly should have done typedef for grids etc...
        typedef std::vector<std::vector<float>> AffineMatrix;

        AffineMatrix createDefaultAffineMatrix() {
            return AffineMatrix(4, std::vector<float>(4, 0.0f)); // Initialize a 4x4 rotation matrix with 0.0f in all elements
        }
        
        // Default bottom row of affine matrix (0, 0, 0, 1) constant !!!
        const std::vector<float> BottomAffineVector = { 0.f, 0.f, 0.f, 1.f };

        // Affine matrix storage
        AffineMatrix targetAffineMatrix; // Target(ADC/DWI) Affine Matrix, not using createDefaultAffineMatrix because they vectors are push_back instead
        AffineMatrix flairAffineMatrix; // FLAIR Affine Matrix
        
        // Inverse
        AffineMatrix targetAffineInverseMatrix = createDefaultAffineMatrix();
        AffineMatrix finalAffineMatrix = createDefaultAffineMatrix(); 

        // Inverses affine matrix (or well, any 4x4 matrix)
        bool inverseAffine(const std::vector<std::vector<float>>& mat, std::vector<std::vector<float>>& result);
        // Multiplies two affine matrices (or well, any 4x4 matrices)
        void matrixAffineMultiplication(const AffineMatrix& mat1, const AffineMatrix& mat2, AffineMatrix& resultMat);

        // Apply matrix to a point
        std::vector<float> applyMatToPoint(const AffineMatrix& mat, float z, float y, float x);
        
        // Uses above function and trilinear interpolation to apply to grid
        bool applyMatToGrid(std::vector<std::vector<std::vector<float>>>& grid, std::vector<std::vector<std::vector<float>>>& result);

        
        void activateSigmoidOverChannels(std::vector<std::vector<std::vector<std::vector<float>>>>& inputChannels);
        void activateReLUOverChannels(std::vector<std::vector<std::vector<std::vector<float>>>>& inputChannels);

        void pool(const std::vector<std::vector<std::vector<std::vector<float>>>>& inputChannels, std::vector<std::vector<std::vector<std::vector<float>>>>& outputChannels, int poolWidth, int poolHeight, int poolDepth, int stride);
        
        void binarySegmentation(const std::vector<std::vector<std::vector<float>>>& inputGrid, std::vector<std::vector<std::vector<float>>>& outputGrid);
        float compLoss(const std::vector<std::vector<std::vector<float>>>& pGrid, const std::vector<std::vector<std::vector<float>>>& tGrid, float smooth, std::vector<std::vector<std::vector<float>>>& gGrid);
        void upsample(const std::vector<std::vector<std::vector<float>>>& inputGrid, std::vector<std::vector<std::vector<float>>>& outputGrid);
        
        // Backpropagation functions
        void backwardUpsample(const std::vector<std::vector<std::vector<float>>>& outputGrad,
            std::vector<std::vector<std::vector<TriLerpCache>>>& CacheGrid, 
            std::vector<std::vector<std::vector<float>>>& inputGrad);
        void backwardSigmoid(const std::vector<std::vector<std::vector<float>>>& outputGrad, 
            std::vector<std::vector<std::vector<float>>>& inputGrad);
        void backwardReLU(const std::vector<std::vector<std::vector<std::vector<float>>>>& outputGrad, 
            const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardInput, 
            std::vector<std::vector<std::vector<std::vector<float>>>>& inputGrad);
        bool backwardConvolve(const std::vector<std::vector<std::vector<std::vector<float>>>>& outputGrad, 
            std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filters, 
            std::vector<std::vector<std::vector<std::vector<float>>>>& inputGrad, 
            std::vector<float>& bias, 
            const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardInput, 
            const float learningRate, 
            const int padding);
        void backwardPool(const std::vector<std::vector<std::vector<std::vector<float>>>>& outputGrad, 
            const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardInput, 
            const std::vector<std::vector<std::vector<std::vector<float>>>>& forwardOutput, 
            std::vector<std::vector<std::vector<std::vector<float>>>>& inputGrad, 
            int poolDepth, int poolHeight, int poolWidth, 
            int stride);
        
        // Gradients calculated in backpropagation
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

        // Exporting to NifTI
        void const gridToNifti(const std::vector<std::vector<std::vector<float>>>& grid, const std::string& filename, const bool bin);




        
    };
}

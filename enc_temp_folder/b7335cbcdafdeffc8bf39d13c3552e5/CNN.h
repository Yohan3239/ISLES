#pragma once
#include <vector>
#include <variant>
#include <string>

namespace C
{
    
    class CNN {

    public:

        bool bIsFlair;

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
        }; // 348 byte struct to read Header data into // Struct to put header data into

        float relu(float x); // RELU

        void readNiftiHeader(const std::string& filename, bool bFlair); // Read Header data

        void convert1To3(std::vector<float>& voxels); //1D to 3D data

        void normalise(std::vector<std::vector<std::vector<float>>>& grid); // Normalise between [,0,1]
        void clear(); // Clear all values for next case

        // Convolutional Layer
        void convolve( 
            const std::vector<std::vector<std::vector<std::vector<float>>>>& input, // Input tensor
            const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filters, // Filters
            std::vector<std::vector<std::vector<std::vector<float>>>>& output, // Output tensor
            int stride
        );

        void initialiseFilter(std::vector<std::vector<std::vector<std::vector<float>>>>& filter,
            int filter_channels, int filter_height, int filter_width, int filter_depth);
        void insertGrid(const std::vector<std::vector<std::vector<float>>>& grid);

        void process16NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
        void process64NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
    private:
        int width; // Resets for each file
        int height; // Same as above
        int depth; // Same as above

        int resultWidth, resultHeight, resultDepth = 100; // What the dimensions should be resampled to. Updated from the resultant file.
        
    public:
        std::vector<std::vector<std::vector<std::vector<float>>>> filter;
        std::vector<std::vector<std::vector<float>>> voxelsGrid; // Input data, dimension FLAIR
        std::vector<std::vector<std::vector<float>>> transformGrid; // Input data, transformed AND resampled using affine matrix, dimension ADC/DWI .
        std::vector<std::vector<std::vector<std::vector<float>>>> gridChannels;
        std::vector<std::vector<std::vector<float>>> convolveGrid; // Output through Convolutional layer

        // TRANSFORM STUFF //
        // Affine Transformation Matrix (4x4)
        typedef std::vector<std::vector<float>> AffineMatrix;

        // Rotation Transformation Matrix (3x3)
        typedef std::vector<std::vector<float>> RotationMatrix;

        RotationMatrix createDefaultRotationMatrix() {
            return RotationMatrix(3, std::vector<float>(3, 0.0f)); // Initialize a 3x3 rotation matrix with 0.0f in all elements
        }
        RotationMatrix createDefaultAffineMatrix() {
            return RotationMatrix(4, std::vector<float>(4, 0.0f)); // Initialize a 4x4 rotation matrix with 0.0f in all elements
        }
        
        // Quaternion structure
        struct Quaternion {
            float a, b, c, d;
        };
        
        // Default bottom row of affine matrix (0, 0, 0, 1) constant !!!

        const std::vector<float> BottomAffineVector = { 0.f, 0.f, 0.f, 1.f };
        
        // Actual quaternions
        Quaternion targetQuaternion;
        Quaternion flairQuaternion;

        // Quaternion->Rotation matrix
        RotationMatrix targetRotMatrix = createDefaultRotationMatrix(); // Target(ADC/DWI) Rotation Matrix
        RotationMatrix flairRotMatrix = createDefaultRotationMatrix(); // FLAIR Rotation Matrix

        // Affine matrix storage
        AffineMatrix targetAffineMatrix; // Target(ADC/DWI) Affine Matrix
        AffineMatrix flairAffineMatrix; // FLAIR Affine Matrix
        
        // Inversed rotation matrix
        RotationMatrix targetRotInverseMatrix = createDefaultRotationMatrix(); // Target(ADC/DWI) Rotation matrix INVERSED to
        AffineMatrix targetAffineInverseMatrix = createDefaultAffineMatrix();

        RotationMatrix finalRotMatrix = createDefaultRotationMatrix();
        AffineMatrix finalAffineMatrix = createDefaultAffineMatrix(); 

        /// <summary>
        /// FLAIR quaternion -> FLAIR rotation matrix.
        /// ADC/DWI quaternion -> ADC/DWI rotation matrix -> ADC/DWI inverse rotation matrix
        /// FLAIR affine matrix
        /// ADC/DWI affine matrix -> ADC/DWI inverse affine matrix
        /// 
        /// Apply (FLAIR rotation mat * ADC/DWI inverse rotation mat) to FLAIR grid
        /// Apply (FLAIR affine mat * ADC/DWI inverse affine mat)
        /// 
        /// ...
        /// 
        /// Resample
        /// </summary>

        void quaternionToMatrix(const Quaternion& quaternion, RotationMatrix& matrix);
        bool inverseRot(const std::vector<std::vector<float>>& mat, std::vector<std::vector<float>>& result);
        void matrixRotMultiplication(const CNN::RotationMatrix mat1, const CNN::RotationMatrix mat2, CNN::RotationMatrix resultMat);

        bool inverseAffine(const std::vector<std::vector<float>>& mat, std::vector<std::vector<float>>& result);
        void matrixAffineMultiplication(const AffineMatrix mat1, const AffineMatrix mat2, AffineMatrix resultMat);

        std::vector<float> applyAffineToPoint(const AffineMatrix mat, float x, float y, float z);
        void applyAffineToGrid(const std::vector<std::vector<std::vector<float>>>& grid, std::vector<std::vector<std::vector<float>>>& result);

    };
}

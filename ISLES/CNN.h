#pragma once
#include <vector>
#include <variant>
#include <string>

namespace C
{
    
    class CNN {

    public:

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

        float relu(float x);
        void readNiftiHeader(const std::string& filename, bool bResample);

        void convert1To3(std::vector<float>& voxels);
        void printVoxelsGrid();

        void normalise();
        void clear();
        void convolve(
            const std::vector<std::vector<std::vector<std::vector<float>>>>& input, // Input tensor
            const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>& filters, // Filters
            std::vector<std::vector<std::vector<std::vector<float>>>>& output, // Output tensor
            int stride
        );
        void initialiseFilter(std::vector<std::vector<std::vector<std::vector<float>>>>& filter,
            int filter_channels, int filter_height, int filter_width, int filter_depth);
        void insertGrid(const std::vector<std::vector<std::vector<float>>>& grid);

        int width; // Resets for each file
        int height; // Same as above
        int depth; // Same as above

        int resultWidth, resultHeight, resultDepth = 100; // What the dimensions should be resampled to. Updated from the resultant file.

        std::vector<std::vector<std::vector<std::vector<float>>>> filter;
        std::vector<std::vector<std::vector<float>>> voxelsGrid; // Input data, normalised.
        std::vector<std::vector<std::vector<std::vector<float>>>> gridChannels;
        std::vector<std::vector<std::vector<float>>> convolveGrid; // Output through Convolutional layer

        void process16NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
        void process64NiftiData(const std::string& filename, int numVoxels, float vox_offset, float scl_slope, float scl_inter, int bitpix);
    };
}

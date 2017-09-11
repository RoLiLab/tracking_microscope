#ifndef HDF5IMAGEWRITER_H
#define HDF5IMAGEWRITER_H

#include <string>
#include <H5Cpp.h>
#include <H5Exception.h>
#include "hdf5.h"
#include "hdf5_hl.h"

using namespace std;
using namespace H5;

#define harddrivecount_FLRsave 5
#define harddrivecount_NIRsave 3


enum{IMAGE_UINT8, IMAGE_UINT16};

class HDF5ImageWriter
{
public:
	HDF5ImageWriter(const string hdf5_filename, const int image_width, const int image_height, int imgbitMode, int chunkImgCount);
	void write(const uint8_t* image);
	void write(const uint16_t* image);
	void flush(void);
    ~HDF5ImageWriter();

private:

    H5File* _file;
    DataSet* _dataset;
    DataSpace* _memory_dataspace;
    hsize_t _write_size[3];
    hsize_t _write_offset[3];
    hsize_t _total_size[3];
};



#endif // HDF5IMAGEWRITER_H

#ifndef HDF5IMAGEREADER_H
#define HDF5IMAGEREADER_H
#include <string>
#include <H5Cpp.h>

using namespace std;
using namespace H5;

class HDF5ImageReader
{
public:
    HDF5ImageReader(const char * hdf5_filename);
    int image_width() const;
	int image_height() const;
	int image_szbyte() const;
    int n_images() const;
	void read(int64_t image_index, uint8_t* image);
	void read16(int64_t image_index, uint16_t* image);
    ~HDF5ImageReader();

private:
    H5File* _file;
    DataSpace* _file_dataspace;
    DataSet* _dataset;
    DataSpace* _memory_dataspace;
    hsize_t _read_size[3];
    hsize_t _read_offset[3];
    hsize_t _total_size[3];
};
#endif // HDF5IMAGEREADER_H

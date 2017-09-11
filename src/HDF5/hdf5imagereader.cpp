#include "Base/base.h"
#include "hdf5imagereader.h"

HDF5ImageReader::HDF5ImageReader(const char * hdf5_filename)
//    : _read_offset{}
{
	try {
		_read_offset[0] = 0;
		_read_offset[1] = 0;
		_read_offset[2] = 0;
		_file = new H5File (hdf5_filename, H5F_ACC_RDONLY);
		_dataset = new DataSet(_file->openDataSet("images"));
		_file_dataspace = new DataSpace(_dataset->getSpace());
		auto rank = _file_dataspace->getSimpleExtentNdims();
		//Q_ASSERT(rank == 3);
		_file_dataspace->getSimpleExtentDims(_total_size);
		_read_size[0] = 1;
		_read_size[1] = image_height();
		_read_size[2] = image_width();
		_memory_dataspace = new DataSpace(3, _read_size);
	}
	catch (H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5ImageReaderError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "HDF5ImageReader function: %s\n", e.getCDetailMsg());
		fclose(ofp);
	} // end catch error
}
int HDF5ImageReader::image_szbyte() const
{
	return _total_size[1] * _total_size[2] * sizeof(uint16);
}

int HDF5ImageReader::image_width() const
{
    return _total_size[2];
}

int HDF5ImageReader::image_height() const
{
    return _total_size[1];
}

int HDF5ImageReader::n_images() const
{
    return _total_size[0];
}

void HDF5ImageReader::read(int64_t image_index, uint8_t *image)
{
    //Q_ASSERT(image_index >= 0 && image_index < n_images());
    _read_offset[0] = image_index;
    _file_dataspace->selectHyperslab(H5S_SELECT_SET, _read_size, _read_offset);
    _dataset->read(image, PredType::NATIVE_UINT8, *_memory_dataspace, *_file_dataspace);
}

void HDF5ImageReader::read16(int64_t image_index, uint16_t *image)
{
	//Q_ASSERT(image_index >= 0 && image_index < n_images());
	_read_offset[0] = image_index;
	_file_dataspace->selectHyperslab(H5S_SELECT_SET, _read_size, _read_offset);
	_dataset->read(image, PredType::NATIVE_UINT16, *_memory_dataspace, *_file_dataspace);
}

HDF5ImageReader::~HDF5ImageReader()
{
    delete _file_dataspace;
    delete _memory_dataspace;
    delete _dataset;
    delete _file;
}

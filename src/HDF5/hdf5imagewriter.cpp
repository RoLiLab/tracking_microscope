#include "Base/base.h"
#include "hdf5imagewriter.h"


HDF5ImageWriter::HDF5ImageWriter(const string hdf5_filename, const int image_width, const int image_height, int imgbitMode, int chunkImgCount)
{
	_write_size[0] = 1;
	_write_size[1] = (hsize_t)image_height;
	_write_size[2] = (hsize_t)image_width;

	_write_offset[0] = 0;
	_write_offset[1] = 0;
	_write_offset[2] = 0;

	_total_size[0] = 0;
	_total_size[1] = (hsize_t)image_height;
	_total_size[2] = (hsize_t)image_width;
	_file = NULL;
	_dataset = NULL;
	_memory_dataspace = NULL;
	try {
	 _file = new H5File(hdf5_filename.c_str(), H5F_ACC_TRUNC);
	}
	catch (H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5ImageWriterError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "HDF5ImageWriter function: %s\n", e.getCDetailMsg());
		fclose(ofp);
	} // end catch error
	//_file = new H5File(hdf5_filename, H5F_ACC_CREAT);

    DSetCreatPropList plist;
    const hsize_t chunk_dims[] = {1, (hsize_t)image_height, (hsize_t)image_width};
    plist.setChunk(3, chunk_dims);
    //plist.setSzip(H5_SZIP_NN_OPTION_MASK, 16);

    const hsize_t file_max_dims[] = {H5S_UNLIMITED, (hsize_t)image_height, (hsize_t)image_width};
    auto file_dataspace = DataSpace(3, _total_size, file_max_dims);

	const hsize_t memory_dataspace_dims[] = { chunkImgCount, (hsize_t)image_height, (hsize_t)image_width };
    _memory_dataspace = new DataSpace (3, memory_dataspace_dims);

	switch (imgbitMode) {
	case IMAGE_UINT8:
		_dataset = new DataSet (_file->createDataSet("images", PredType::NATIVE_UINT8, file_dataspace, plist) );
		break;
	case IMAGE_UINT16:
		_dataset = new DataSet (_file->createDataSet("images", PredType::NATIVE_UINT16, file_dataspace, plist) );
		break;
	default:
		_dataset = new DataSet (_file->createDataSet("images", PredType::NATIVE_UINT8, file_dataspace, plist) );
		break;
	}
}

void HDF5ImageWriter::write(const uint8_t *image)
{
	if (_dataset) {
		_total_size[0]++;
		_dataset->extend(_total_size);
		DataSpace file_dataspace ( _dataset->getSpace() );
		file_dataspace.selectHyperslab(H5S_SELECT_SET, _write_size, _write_offset);
		_dataset->write(image, PredType::NATIVE_UINT8, *_memory_dataspace, file_dataspace);
		_write_offset[0]++;
	}
}

void HDF5ImageWriter::write(const uint16_t *image)
{
	if (_dataset) {
		_total_size[0]++;
		_dataset->extend(_total_size);
		DataSpace file_dataspace ( _dataset->getSpace() );
		file_dataspace.selectHyperslab(H5S_SELECT_SET, _write_size, _write_offset);
		_dataset->write(image, PredType::NATIVE_UINT16, *_memory_dataspace, file_dataspace);
		_write_offset[0]++;
	}
}
void HDF5ImageWriter::flush(void)
{
	if (_file) {
		_file->flush(H5F_SCOPE_GLOBAL);
	}
}
HDF5ImageWriter::~HDF5ImageWriter()
{
	if (_memory_dataspace)
		delete _memory_dataspace;
	if (_dataset)
		delete _dataset;
	if (_file)
		delete _file;
}

#include "Base/base.h"
#include "HDF5\hdf5_TrackingMicroscope.h"




HDF5PointDataSet::HDF5PointDataSet(H5File* _file, char * Name, int SingleData_Type)
{
	_write_size[0] = 1;
	_write_size[1] = 2;

	_write_offset[0] = 0;
	_write_offset[1] = 0;
	_write_offset[2] = 0;

	_total_size[0] = 0;
	_total_size[1] = 2;

	_dataset = NULL;
	_memory_dataspace = NULL;

    DSetCreatPropList plist;
    const hsize_t chunk_dims[] = {1024, 2};
    plist.setChunk(2, chunk_dims);
    //plist.setSzip(H5_SZIP_NN_OPTION_MASK, 16);

    const hsize_t file_max_dims[] = {H5S_UNLIMITED, 2};
    auto file_dataspace = DataSpace(2, _total_size, file_max_dims);

    const hsize_t memory_dataspace_dims[] = {1, 2};
    _memory_dataspace = new DataSpace (2, memory_dataspace_dims);

	_DataType = SingleData_Type;

    _dataset = new DataSet (_file->createDataSet(Name, convertDatatype2H5PredType(_DataType), file_dataspace, plist) );
}

void HDF5PointDataSet::write(void * point)
{
	//try {
		double temp2d[2] = { 0, 0 };
		float temp2f[2] = { 0, 0 };
		int temp2i[2] = { 0, 0 };
		if (_dataset) {
			_total_size[0]++;
			_dataset->extend(_total_size);
			DataSpace file_dataspace(_dataset->getSpace());
			file_dataspace.selectHyperslab(H5S_SELECT_SET, _write_size, _write_offset);
			switch (_DataType) { // switch: get the key (1-9 : number pad)
			case H5_UINT:
				_dataset->write((uint*)point, PredType::NATIVE_UINT, *_memory_dataspace, file_dataspace);
				break;
			case H5_double:
				_dataset->write((double*)point, PredType::NATIVE_DOUBLE, *_memory_dataspace, file_dataspace);
				break;
			case H5_float:
				_dataset->write((float*)point, PredType::NATIVE_FLOAT, *_memory_dataspace, file_dataspace);
				break;
			case H5_UINT64:
				_dataset->write((uint64*)point, PredType::NATIVE_UINT64, *_memory_dataspace, file_dataspace);
			case H5_INT:
				_dataset->write((int*)point, PredType::NATIVE_INT, *_memory_dataspace, file_dataspace);
				break;
			case H5_doubleCV:
				temp2d[0] = ((Point2d *)point)->x;
				temp2d[1] = ((Point2d *)point)->y;
				_dataset->write((double*)temp2d, PredType::NATIVE_DOUBLE, *_memory_dataspace, file_dataspace);
				break;
			case H5_floatCV:
				temp2f[0] = ((Point2f *)point)->x;
				temp2f[1] = ((Point2f *)point)->y;
				_dataset->write((float*)temp2f, PredType::NATIVE_FLOAT, *_memory_dataspace, file_dataspace);
				break;
			case H5_intCV:
				temp2i[0] = ((Point2i *)point)->x;
				temp2i[1] = ((Point2i *)point)->y;
				_dataset->write((uint64*)temp2i, PredType::NATIVE_INT, *_memory_dataspace, file_dataspace);
				break;
			} // end switch

			_write_offset[0]++;
		}
	//}
	//catch (H5::Exception  & e) {
	//	char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5PointDataSetWriteError.txt";
	//	FILE  * ofp = fopen(outputFilename, "w");
	//	fprintf(ofp, "HDF5PointDataSet write function: %s\n", e.getCDetailMsg());
	//	fclose(ofp);
	//} // end catch error
}

HDF5PointDataSet::~HDF5PointDataSet()
{
	if (_memory_dataspace)
		delete _memory_dataspace;
	if (_dataset)
		delete _dataset;
}

//!--------------------------------------------------------------------------------

HDF5PointDataSetReader::HDF5PointDataSetReader(H5File* _file, char * Name, int SingleData_Type)
{
	try {
		_read_offset[0] = 0;
		_read_offset[1] = 0;
		_dataset = new DataSet(_file->openDataSet(Name));
		_file_dataspace = new DataSpace(_dataset->getSpace());
		auto rank = _file_dataspace->getSimpleExtentNdims();
		_file_dataspace->getSimpleExtentDims(_total_size);
		_read_size[0] = 1;
		_read_size[1] = 2;
		_memory_dataspace = new DataSpace(2, _read_size);
		_DataType = SingleData_Type;
	}
	catch(H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5PointDataSetReaderError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "HDF5PointDataSetReader function: %s\n", e.getCDetailMsg());
		fclose(ofp);
	} // end catch error
}



int HDF5PointDataSetReader::Data_Length(void)
{
    return _total_size[0];
}


void HDF5PointDataSetReader::read(int64_t image_index, void * point)
{

    //Q_ASSERT(image_index >= 0 && image_index < n_images());
    _read_offset[0] = image_index;
    _file_dataspace->selectHyperslab(H5S_SELECT_SET, _read_size, _read_offset);
	double temp2d[2] = {0};
	float temp2f[2] = {0};
	uint64 temp2i[2] = {0};

	switch(_DataType) { // switch: get the key (1-9 : number pad)
	case H5_UINT:
		_dataset->read((uint*)point, PredType::NATIVE_UINT, *_memory_dataspace, *_file_dataspace);
		break;
	case H5_double:
		_dataset->read((double*)point, PredType::NATIVE_DOUBLE, *_memory_dataspace, *_file_dataspace);
		break;
	case H5_float:
		_dataset->read((float*)point, PredType::NATIVE_FLOAT, *_memory_dataspace, *_file_dataspace);
		break;
	case H5_UINT64:
		_dataset->read((uint64*)point, PredType::NATIVE_UINT64, *_memory_dataspace, *_file_dataspace);
	case H5_INT:
		_dataset->read((int*)point, PredType::NATIVE_INT, *_memory_dataspace, *_file_dataspace);
		break;
	case H5_doubleCV:
		_dataset->read((double*)temp2d, PredType::NATIVE_DOUBLE, *_memory_dataspace, *_file_dataspace);
		((Point2d *)point)->x = temp2d[0];
		((Point2d *)point)->y = temp2d[1];
		break;
	case H5_floatCV:
		_dataset->read((float*)temp2f, PredType::NATIVE_FLOAT, *_memory_dataspace, *_file_dataspace);
		((Point2f *)point)->x = temp2f[0];
		((Point2f *)point)->y = temp2f[1];
		break;
	case H5_intCV:
		_dataset->read((uint64*)temp2i, PredType::NATIVE_INT, *_memory_dataspace, *_file_dataspace);
		((Point2f *)point)->x = temp2i[0];
		((Point2f *)point)->y = temp2i[1];
		break;
	} // end switch
}

HDF5PointDataSetReader::~HDF5PointDataSetReader()
{
	if (_file_dataspace)
		delete _file_dataspace;
	if (_memory_dataspace)
		delete _memory_dataspace;
	if (_dataset)
		delete _dataset;
}

//!--------------------------------------------------------------------------------

HDF5SingleDataSet::HDF5SingleDataSet(H5File* _file, char * Name, int SingleData_Type)
{
	try {
		_write_size = 1;

		_write_offset = 0;

		_total_size = 0;

		_dataset = NULL;
		_memory_dataspace = NULL;

		DSetCreatPropList plist;
		const hsize_t chunk_dims[] = { 1024 };
		plist.setChunk(1, chunk_dims);
		//plist.setSzip(H5_SZIP_NN_OPTION_MASK, 16);

		const hsize_t file_max_dims[] = { H5S_UNLIMITED };
		auto file_dataspace = DataSpace(1, &_total_size, file_max_dims);

		const hsize_t memory_dataspace_dims[] = { 1 };
		_memory_dataspace = new DataSpace(1, memory_dataspace_dims);

		_DataType = SingleData_Type;

		_dataset = new DataSet(_file->createDataSet(Name, convertDatatype2H5PredType(_DataType), file_dataspace, plist));
	}
	catch (H5::Exception  & e) {
		//char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5SingleDataSetWriteError.txt";
		//FILE  * ofp = fopen(outputFilename, "w");
		//fprintf(ofp, "HDF5SingleDataSet write function: %s\n", e.getCDetailMsg());
		//fclose(ofp);
		//int a = 1;
	} // end catch error
}

void HDF5SingleDataSet::write(void * data)
{
	//try {
		if (_dataset) {
			_total_size++;
			_dataset->extend(&_total_size);
			DataSpace file_dataspace(_dataset->getSpace());
			file_dataspace.selectHyperslab(H5S_SELECT_SET, &_write_size, &_write_offset);

			switch (_DataType) { // switch: get the key (1-9 : number pad)
			case H5_UINT:
				_dataset->write((unsigned int*)data, PredType::NATIVE_UINT, *_memory_dataspace, file_dataspace);
				break;
			case H5_double:
				_dataset->write((double*)data, PredType::NATIVE_DOUBLE, *_memory_dataspace, file_dataspace);
				break;
			case H5_float:
				_dataset->write((float*)data, PredType::NATIVE_FLOAT, *_memory_dataspace, file_dataspace);
				break;
			case H5_UINT64:
				_dataset->write((uint64*)data, PredType::NATIVE_UINT64, *_memory_dataspace, file_dataspace);
				break;
			case H5_INT:
				_dataset->write((int*)data, PredType::NATIVE_INT, *_memory_dataspace, file_dataspace);
				break;
			} // end switch

			_write_offset++;
		}
	//}
	//catch (H5::Exception  & e) {
	//	char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5SingleDataSetWriteError.txt";
	//	FILE  * ofp = fopen(outputFilename, "w");
	//	fprintf(ofp, "HDF5SingleDataSet write function: %s\n", e.getCDetailMsg());
	//	fclose(ofp);
	//} // end catch error
}

HDF5SingleDataSet::~HDF5SingleDataSet()
{
	if (_memory_dataspace)
		delete _memory_dataspace;
	if (_dataset)
		delete _dataset;
}

//!--------------------------------------------------------------------------------

HDF5SingleDataSetReader::HDF5SingleDataSetReader(H5File* _file, char * Name, int SingleData_Type)
{
	try {
		_read_offset = 0;
		_dataset = new DataSet(_file->openDataSet(Name));
		_file_dataspace = new DataSpace(_dataset->getSpace());
		auto rank = _file_dataspace->getSimpleExtentNdims();
		_file_dataspace->getSimpleExtentDims(&_total_size);
		_read_size = 1;
		_memory_dataspace = new DataSpace(1, &_read_size);
		_DataType = SingleData_Type;
	}
	catch (H5::Exception  & e) {
		char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\HDF5SingleDataSetReaderError.txt";
		FILE  * ofp = fopen(outputFilename, "w");
		fprintf(ofp, "HDF5SingleDataSetReader function: %s\n", e.getCDetailMsg());
		fclose(ofp);
	} // end catch error
}



int HDF5SingleDataSetReader::Data_Length(void)
{
    return _total_size;
}


void HDF5SingleDataSetReader::read(int64_t index, void * point)
{
	//Q_ASSERT(image_index >= 0 && image_index < n_images());
    _read_offset = index;
    _file_dataspace->selectHyperslab(H5S_SELECT_SET, &_read_size, &_read_offset);
	double temp2d= 0;
	float temp2f = 0;
	uint64 temp2i = 0;

	switch(_DataType) { // switch: get the key (1-9 : number pad)
	case H5_UINT:
		_dataset->read((uint*)point, PredType::NATIVE_UINT, *_memory_dataspace, *_file_dataspace);
		break;
	case H5_double:
		_dataset->read((double*)point, PredType::NATIVE_DOUBLE, *_memory_dataspace, *_file_dataspace);
		break;
	case H5_float:
		_dataset->read((float*)point, PredType::NATIVE_FLOAT, *_memory_dataspace, *_file_dataspace);
		break;
	case H5_UINT64:
		_dataset->read((uint64*)point, PredType::NATIVE_UINT64, *_memory_dataspace, *_file_dataspace);
	case H5_INT:
		_dataset->read((int*)point, PredType::NATIVE_INT, *_memory_dataspace, *_file_dataspace);
		break;
	} // end switch
}

HDF5SingleDataSetReader::~HDF5SingleDataSetReader()
{
	if (_file_dataspace)
		delete _file_dataspace;
	if (_memory_dataspace)
		delete _memory_dataspace;
	if (_dataset)
		delete _dataset;
}

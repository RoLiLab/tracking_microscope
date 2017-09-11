#ifndef HDF5DATAREADER_H
#define HDF5DATAREADER_H
#include <string>
#include <H5Cpp.h>
#include "HDF5\hdf5_TrackingMicroscope.h"

using namespace std;
using namespace H5;

class HDF5DataReader
{
public:
    HDF5DataReader(const char * hdf5_filename);
    ~HDF5DataReader();
	
    vector<HDF5SingleDataSetReader *> SingleData;
	vector<HDF5PointDataSetReader *> PointData;

private:
    H5File* _file;
};

#endif // HDF5IMAGEREADER_H

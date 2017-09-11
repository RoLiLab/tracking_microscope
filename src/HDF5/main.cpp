#include <iostream>
#include <cstdint>
#include <QDebug>
#include <QVector>
#include <QImage>
#include <QTime>
#include <H5Cpp.h>
#include "hdf5imagewriter.h"
#include "hdf5imagereader.h"

using namespace std;
using namespace H5;

int main(int, char**)
{
    QImage* images[4485];
    for (auto i = 0; i < 4485; i++) {
        auto image = new QImage();
        image->load(("/Users/drobson/Desktop/zebrafish_images.pngfolder/Frame" + to_string(i) + ".png").c_str(), "PNG");
        images[i] = image;
    }

    HDF5ImageWriter writer("/Users/drobson/Desktop/zebrafish_images.pngfolder/images.h5", 659, 494);
    for (int i = 0; i < 4485; i++)
        writer.write(images[i]->bits());

    HDF5ImageReader reader("/Users/drobson/Desktop/zebrafish_images.pngfolder/images.h5");
    uint8_t image_data[reader.image_width() * reader.image_height()];
    for (int i = 0; i < 4485; i++)
        reader.read(i, image_data);
}

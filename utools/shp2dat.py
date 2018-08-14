'''
NAME
    Convert shapefile to ASCII
PURPOSE
    To to read shapefiles and convert to ASCII formats

PROGRAMMER(S)
    Bogdan Hlevca
REVISION HISTORY
    20161201 -- Initial version created

REFERENCES

'''

import numpy as np
import os
import sys
import shapefile

ShapeType = {
    0: "NULL",
    1: "POINT",
    3: "POLYLINE",
    5: "POLYGON",
    6: "MULTIPOINT",
    11: "POINTZ",
    13: "POLYLINEZ",
    15: "POLYGONZ",
    18: "MULTIPOINTZ",
    21: "POINTM",
    23: "POLYLINEM",
    25: "POLYGONM",
    28: "MULTIPOINTM",
    31: "MULTIPATCH"
}

def shp2ascii(shpfilename):
    asc_shapes = []
    if shpfilename is not None:
        reader = shapefile.Reader(shpfilename)
        print("Shape Type: %s" % (ShapeType[reader.shapeType]))

        for shape in reader.shapeRecords():
            shp = []
            print("Shape ID: %d, Type: %s" % (shape.record[0], ShapeType[shape.shape.shapeType]))
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            shp = [shape.record[0], x, y]
            asc_shapes.append(shp)
    else:
        print("file not found!")
    return asc_shapes

def write_ascii(fname, asc_shapes):
    f = open(fname, 'w+')
    for shape in asc_shapes:
        id = "<\n"
        print("[ID:%d]\n" % shape[0])
        f.write(id)
        for x,y in zip(shape[1], shape[2]):
            coord = str(x) + " " + str(y) + "\n"
            f.write(coord)
    f.close()

if __name__ == '__main__':
    asc_shapes = shp2ascii("/home/bogdan/outputshp/coastal_line_Merge.shp")
    write_ascii("/home/bogdan/outputshp/BH_coastal_line.dat", asc_shapes)
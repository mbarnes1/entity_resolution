#/usr/bin/python

import cPickle as pickle  # for saving shit
from record import Record
import numpy as np


# .csv records are of the format posterid,site,Date,State,City,Perspective_1st,Perspective_3rd,
# Name,Age,Cost,Height_ft,Height_in,Weight,Cup,Chest,Waist,Hip,Ethnicity,SkinColor,EyeColor,
# HairColor,Restriction_Type,Restriction_Ethnicity,Restriction_Age,PhoneNumber,AreaCode_State,
# AreaCode_Cities,Email,Url,Media,Images
class RecordDatabase(object):
    def __init__(self, annotation_path, max_records=np.Inf):
        self._annotation_path = annotation_path
        self.records = dict()
        self._max_records = max_records
        #print 'Opening file...'
        ins = open(self._annotation_path, 'r')
        ins.next()
        linecounter = 0  # line counter. Zero indexed

        # Loop through all the records
        for sample in ins:
            print 'Extracting sample', linecounter
            r = Record()  # record object from utils
            r.initialize_from_annotation(sample.rstrip('\n'), linecounter)
            self.records[linecounter] = r
            linecounter += 1  # increment node index counter
            if linecounter >= self._max_records:
                break
        #print 'Finished generating database'

    def dump(self, dump_path):
        print 'Pickling record hash table...'
        pickle.dump(self, open(dump_path, 'wb'))
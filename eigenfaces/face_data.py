import numpy as np
import matplotlib

from PIL import Image

class face_data :
    def __init__( self, data ):
        self._data = data.astype( np.float32 )/255.0

        rdata = self._data.ravel()
        self._dim = len( rdata )

        self._vec = np.matrix( self._data.ravel().reshape([-1,1]) )

    @property
    def dim( self ):
        return self._dim

    @property
    def vec( self ):
        return self._vec

def load_data( fname='training.csv' ):
    with open( fname, 'r' ) as ff :
        lines = ff.readlines()

    keys = lines[0][:-1].split(',')

    faces = []

    for l in lines[1:] :
        img = l[:-1].split(',')[-1]
        img = img.split(' ')
        img = np.array([ int(v) for v in img ], dtype=np.uint8).reshape((96,96))

        img_pil = Image.fromarray( img )
        img_pil = img_pil.resize((32,32))

        faces.append( face_data( np.array( img_pil ) ) )

    return np.array( faces )

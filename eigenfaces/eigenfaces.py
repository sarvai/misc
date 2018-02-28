import numpy as np

def _calc_meanface( faces, dim ):
    meanface = np.matrix(np.zeros((dim,1)))
    for face in faces :
        meanface = meanface + face.vec
    meanface = meanface / len( faces )
    return meanface

def _calc_covmat( faces, meanface, dim ):
    ovmat = np.zeros((dim,dim))

    covmat = np.zeros((dim,dim))

    for face in faces :
        v = face.vec - meanface
        covmat = covmat + v * v.T

    covmat = covmat / len(faces)

    return covmat

def calc( faces, k=4 ):
    dim = faces[0].dim

    meanface = _calc_meanface( faces, dim )
    covmat = _calc_covmat( faces, meanface, dim )

    w, v = np.linalg.eig( covmat )

    assert np.max( np.abs( np.imag(w) ) ) < 1e-10, "Eigenvalues are not real"

    w = np.real(w)
    v = np.real(v)

    order = np.argsort( w )[::-1]
    v = v[:,order]

    return w[:k], v[:,:k]

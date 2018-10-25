import sys
from ..models.detector import Detector
from ..models.fcn_detector import FcnDetector
from ..models.mtcnnDetector import MtcnnDetector
from ..models.mtcnn_inference import P_Net, R_Net, O_Net


def mtccn( modelpath =None,
           minfacesize = 24,
           testmodel = 'ONet'
           , **kwargs):
    """
               'mtcnn()' function attempt to build mtcnn model with some initialization parameter.

              :param modelpath: the path of the model's weights
              :param minfacesize: the minimum of the face size to be detect
              :param testmodel : default is 'ONet' and do not change it normally.
              :return: mtcnn model
            """
    min_face_size = minfacesize
    batchsize = [1, 64, 8]
    thresh = [0.9, 0.6, 0.7]
    detectors = [None, None, None]

    detectors[0] = FcnDetector(P_Net, modelpath[0])
    detectors[1] = Detector(R_Net, 24, batchsize[1], modelpath[1]) if testmodel in ["RNet" ,"ONet"] else None
    detectors[2] = Detector(O_Net, 48, batchsize[2], modelpath[2]) if testmodel in ["ONet"] else None

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                    threshold=thresh)
    return mtcnn_detector
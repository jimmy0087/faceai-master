from faceai.Detection import *
from .dan_models import *


def dan(modelpath =None,
        **kwargs):
    """
       'dan()' function attempt to build DAN model with some initialization parameter.

       :param modelpath: the path of the model's weights
       :return: DAN model
    """
    execution_path = os.path.dirname(__file__)
    initInf = np.load(os.path.join(execution_path,'initInf.npz'))

    danDetactor = DANDetector(initInf, modelpath)
    return danDetactor

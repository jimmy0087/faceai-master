import os
import cv2
import numpy as np
import tensorflow as tf
from .DAN.models.DAN import dan
from .PRNet.models.api import PRN
from ..Utils.images import read_image_bgr, read_image_array, preprocess_image, resize_image
from ..Utils.visualization import draw_box, draw_caption,draw_landmarks
from ..Utils.download import download_file_from_google_drive



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

class LandmarksDetection:
    """
        This is the face landmarks class for images in the FaceAI library. It provides support for DAN
         face landmarks detection network . After instantiating this class, you can set it's properties and
         make landmarks detections using it's pre-defined functions.

         The following functions are required to be called before face detection can be made
         * setModelPath()
         * At least of the following and it must correspond to the model set in the setModelPath()
          [setModelTypeAsDAN()]
         * loadModel() [This must be called once only before performing object detection]

         Once the above functions have been called, you can call the detectLandmarksFromImage() function of
         the face detection instance face at anytime to obtain observable objects in any image.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_min = 1333
        self.__input_image_max = 800
        self.__model_id={ 'dan' : '1Qh_OWZROneM01q8fxS9Clf7zfj5FBTdd',
                         'prnet': '1yAb66iD7PF_GHTbXSSndOVlY6xXOJhFt'}

    def setModelTypeAsDAN(self):
        """
        'setModelTypeAsDAN()' is used to set the model type to the DAN model
        for the face detection.
        :return:
        """
        self.__modelType = "dan"

    def setModelTypeAsPRNet(self):
        """
        'setModelTypeAsDAN()' is used to set the model type to the DAN model
        for the face detection.
        :return:
        """
        self.__modelType = "prnet"

    def loadModel(self,model_path=''):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function.
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.

                :param
                :return:
        """
        cache_dir = os.path.join(os.path.expanduser('~'), '.faceai')

        if (self.__modelLoaded == False):
            if(self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif(self.__modelType == "dan"):
                des_file = '/'.join((cache_dir,self.__modelType))
                self.modelPath = download_file_from_google_drive(self.__model_id[self.__modelType], des_file)
                model = dan(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True
            elif (self.__modelType == "prnet"):
                des_file = '/'.join((cache_dir, self.__modelType))
                self.modelPath = download_file_from_google_drive(self.__model_id[self.__modelType], des_file)
                model = PRN(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True


    def detectLandmarksFromImage(self,input_image="",dets=None,
                             points_mark = False):
        """
            'detectLanmarksFromImage()' function is used to detect faces landmarks in the given image path:
                    * input_image , which can be file to path, image numpy array ,the function will recognize the type of the input automatically
                    * dets , the boundaries of the objective faces
                    * points_mark , whether the output image are marked with points

            The values returned by this function as follows:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * landmarks coordinate

            :param input_image:
            :param points_mark:
            :return detected_copy:
            :return output_objects_array:
        """

        if(self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        elif(self.__modelLoaded == True):
            try:
                output_objects_array = []

                if type(input_image) == str:
                    image = read_image_bgr(input_image)
                elif type(input_image) == np.ndarray:
                    image = read_image_array(input_image)
                else:
                    raise ValueError("Wrong type of the input image.")
                detected_copy = image.copy()

                model = self.__model_collection[0]
                for i,det in enumerate(dets):
                    landmarks = model.processImg(image,det).astype(np.int16)

                    if points_mark == True:
                        draw_landmarks(detected_copy, landmarks)

                    each_object_details = {}
                    #each_object_details["name"] = self.numbers_to_names[label]
                    each_object_details["landmarks_details"] = landmarks
                    output_objects_array.append(each_object_details)

                return detected_copy, output_objects_array
            except:
                raise ValueError("Ensure you specified correct input image, input type, input rectangles!")

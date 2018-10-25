import os
import cv2
import numpy as np
import matplotlib.image as pltimage
import tensorflow as tf
from .MTCNN.utils.image import read_image_bgr, read_image_array, read_image_stream, preprocess_image, resize_image
from .MTCNN.models.MTCCN import mtccn
from .MTCNN.utils.visualization import draw_box, draw_caption
from .MTCNN.utils.colors import label_color



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

class FacesDetection:
    """
                    This is the face detection class for images in the FaceAI library. It provides support for MTCCN
                     face detection network . After instantiating this class, you can set it's properties and
                     make object detections using it's pre-defined functions.

                     The following functions are required to be called before face detection can be made
                     * setModelPath()
                     * At least of the following and it must correspond to the model set in the setModelPath()
                      [setModelTypeAsRetinaNet()]
                     * loadModel() [This must be called once only before performing object detection]

                     Once the above functions have been called, you can call the detectFacesFromImage() function of
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

        self.numbers_to_names = {0: 'face'}

    def setModelTypeAsMTCNN(self):
        """
        'setModelTypeAsMTCNN()' is used to set the model type to the MTCNN model
        for the face detection.
        :return:
        """
        self.__modelType = "MTCCN"

    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a the RetinaNet
          face detection model trained on the MIDER FACE dataset.
          :param model_path:
          :return:
        """

        if(self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True

    def loadModel(self, detection_speed="normal",min_face_size = 24):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function. This function receives an optional value which is "detection_speed".
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.


                * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"

                :param detection_speed:
                :return:
        """

        if(detection_speed=="normal"):
            self.__input_image_min = 800
            self.__input_image_max = 1333
        elif(detection_speed=="fast"):
            self.__input_image_min = 400
            self.__input_image_max = 700
        elif(detection_speed=="faster"):
            self.__input_image_min = 300
            self.__input_image_max = 500
        elif (detection_speed == "fastest"):
            self.__input_image_min = 200
            self.__input_image_max = 350
        elif (detection_speed == "flash"):
            self.__input_image_min = 100
            self.__input_image_max = 250


        if (self.__modelLoaded == False):
            if(self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif(self.__modelType == "MTCCN"):
                model = mtccn(self.modelPath,minfacesize=min_face_size)
                self.__model_collection.append(model)
                self.__modelLoaded = True


    def detectFacesFromImage(self, input_image="", output_image_path="", input_type="file", output_type="file", caption = False,extract_detected_objects = False, minimum_percentage_probability = 50):
        """
            'detectObjectsFromImage()' function is used to detect faces observable in the given image path:
                    * input_image , which can be file to path, image numpy array or image file stream
                    * output_image_path (only if output_type = file) , file path to the output image that will contain the detection boxes and label, if output_type="file"
                    * input_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file", "array" and "stream"
                    * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
                    * caption(default is false), option to output image whether have caption
                    * extract_detected_objects (optional) , option to save each object detected individually as an image and return an array of the objects' image path.
                    * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.

            The values returned by this function depends on the parameters parsed. The possible values returnable
            are stated as below
            - If extract_detected_objects = False or at its default value and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name
                    * percentage_probability

            - If extract_detected_objects = False or at its default value and output_type = 'array' ,
              Then the function will return:

                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name
                    * percentage_probability

            - If extract_detected_objects = True and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name
                    * percentage_probability
                2. an array of string paths to the image of each object extracted from the image

            - If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name
                    * percentage_probability
                3. an array of numpy arrays of each object detected in the image


            :param input_image:
            :param output_image_path:
            :param input_type:
            :param output_type:
            :param caption
            :param extract_detected_objects:
            :param minimum_percentage_probability:
            :return output_objects_array:
            :return detected_copy:
            :return detected_detected_objects_image_array:
        """

        if(self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        elif(self.__modelLoaded == True):
            try:
                output_objects_array = []
                detected_objects_image_array = []

                if type(input_image) == str:
                    input_type = "file"
                elif type(input_image) == np.ndarray:
                    input_type = "array"
                else:
                    raise ValueError("Wrong type of the input image.")

                if(input_type=="file"):
                    image = read_image_bgr(input_image)
                elif(input_type=="array"):
                    image = read_image_array(input_image)
                elif(input_type=="stream"):
                    image = read_image_stream(input_image)

                detected_copy = image.copy()
                detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                detected_copy2 = image.copy()
                detected_copy2 = cv2.cvtColor(detected_copy2, cv2.COLOR_BGR2RGB)

                #image = preprocess_image(image)
                image, scale = resize_image(image, min_side=self.__input_image_min, max_side=self.__input_image_max)

                model = self.__model_collection[0]
                detections,time = model.detect(image)
                faces_number = detections.shape[0]
                scores = detections[:,4]
                detections[:, :4] /= scale

                min_probability = minimum_percentage_probability / 100
                counting = 0

                for index, score, in enumerate(scores):
                    if score < min_probability:
                        continue

                    counting += 1

                    objects_dir = output_image_path + "-objects"
                    if(extract_detected_objects == True and output_type=="file"):
                        if (os.path.exists(objects_dir) == False):
                            os.mkdir(objects_dir)
                    detection_details = detections[index, :4].astype(int)
                    draw_box(detected_copy, detection_details, color=label_color(2))
                    if caption == True:
                        caption = "{:.3f}".format((score * 100))
                        draw_caption(detected_copy, detection_details, caption)
                    each_object_details = {}
                    #each_object_details["name"] = self.numbers_to_names[label]
                    each_object_details["percentage_probability"] = str(score * 100)
                    output_objects_array.append(each_object_details)

                    if(extract_detected_objects == True):
                        splitted_copy = detected_copy2.copy()[detection_details[1]:detection_details[3],
                                        detection_details[0]:detection_details[2]]
                        if(output_type=="file"):
                            splitted_image_path = os.path.join(objects_dir, self.numbers_to_names[0] + "-" + str(
                                counting) + ".jpg")
                            pltimage.imsave(splitted_image_path, splitted_copy)
                            detected_objects_image_array.append(splitted_image_path)
                        elif(output_type=="array"):
                            detected_objects_image_array.append(splitted_copy)

                if(output_type=="file"):
                    pltimage.imsave(output_image_path, detected_copy)

                if(extract_detected_objects == True):
                    if(output_type=="file"):
                        return output_objects_array, detected_objects_image_array
                    elif(output_type=="array"):
                        return detected_copy, output_objects_array, detected_objects_image_array

                else:
                    if (output_type == "file"):
                        return output_objects_array
                    elif (output_type == "array"):
                        return detected_copy, output_objects_array
            except:
                raise ValueError("Ensure you specified correct input image, input type, output type and/or output image path ")



class VideoFaceDetection:
    """
                    This is the face detection class for images in the FaceAI library. It provides support for MTCCN
                     face detection network . After instantiating this class, you can set it's properties and
                     make object detections using it's pre-defined functions.

                     The following functions are required to be called before face detection can be made
                     * setModelPath()
                     * At least of the following and it must correspond to the model set in the setModelPath()
                      [setModelTypeAsRetinaNet()]
                     * loadModel() [This must be called once only before performing object detection]

                     Once the above functions have been called, you can call the detectFacesFromImage() function of
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

        self.numbers_to_names = {0: 'face'}

    def setModelTypeAsMTCNN(self):
        """
        'setModelTypeAsMTCNN()' is used to set the model type to the MTCNN model
        for the face detection.
        :return:
        """
        self.__modelType = "MTCCN"

    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a the RetinaNet
          face detection model trained on the MIDER FACE dataset.
          :param model_path:
          :return:
        """

        if(self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True

    def loadModel(self, detection_speed="normal",min_face_size = 24):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function. This function receives an optional value which is "detection_speed".
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.


                * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"

                :param detection_speed:
                :return:
        """

        if(detection_speed=="normal"):
            self.__input_image_min = 800
            self.__input_image_max = 1333
        elif(detection_speed=="fast"):
            self.__input_image_min = 400
            self.__input_image_max = 700
        elif(detection_speed=="faster"):
            self.__input_image_min = 300
            self.__input_image_max = 500
        elif (detection_speed == "fastest"):
            self.__input_image_min = 200
            self.__input_image_max = 350
        elif (detection_speed == "flash"):
            self.__input_image_min = 100
            self.__input_image_max = 250


        if (self.__modelLoaded == False):
            if(self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif(self.__modelType == "MTCCN"):
                model = mtccn(self.modelPath,minfacesize=min_face_size)
                self.__model_collection.append(model)
                self.__modelLoaded = True

    def detectFacesFromVideo(self, input_file_path="", output_file_path="", frames_per_second=20, frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False):

        """
                    'detectObjectsFromVideo()' function is used to detect objects observable in the given video path:
                            * input_file_path , which is the file path to the input video
                            * output_file_path , which is the path to the output video
                            * frames_per_second , which is the number of frames to be used in the output video
                            * frame_detection_interval (optional, 1 by default)  , which is the intervals of frames that will be detected.
                            * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                            * log_progress (optional) , which states if the progress of the frame processed is to be logged to console


                    :param input_file_path:
                    :param output_file_path:
                    :param frames_per_second:
                    :param frame_detection_interval:
                    :param minimum_percentage_probability:
                    :param log_progress:
                    :return output_video_filepath:
                """

        if(input_file_path == ""  or output_file_path ==""):
            raise ValueError("You must set 'input_file_path' to a valid video file, and the 'output_file_path' to path you want the detected video saved.")
        else:
            try:
                input_video = cv2.VideoCapture(input_file_path)
                output_video_filepath = output_file_path + '.avi'

                frame_width = int(input_video.get(3))
                frame_height = int(input_video.get(4))
                output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frames_per_second,
                                               (frame_width, frame_height))

                counting = 0
                predicted_numbers = None
                scores = None
                detections = None

                model = self.__model_collection[0]

                while (input_video.isOpened()):
                    ret, frame = input_video.read()

                    if (ret == True):

                        counting += 1

                        if(log_progress == True):

                            print("Processing Frame : ", str(counting))

                        detected_copy = frame.copy()
                        #detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)
                        frame, scale = resize_image(frame, min_side=self.__input_image_min, max_side=self.__input_image_max)

                        check_frame_interval = counting % frame_detection_interval

                        if(counting == 1 or check_frame_interval == 0):
                            detections, time = model.detect(frame)
                            scores = detections[:, 4]
                            detections[:, :4] /= scale

                        min_probability = minimum_percentage_probability / 100

                        for index, score in enumerate(scores):
                            if score < min_probability:
                                continue
                            detection_details = detections[0, :4].astype(int)
                            draw_box(detected_copy, detection_details, color=label_color(2))

                            caption = "{:.3f}".format((score * 100))
                            draw_caption(detected_copy, detection_details, caption)

                        output_video.write(detected_copy)
                    else:
                        break
                input_video.release()
                output_video.release()

                return output_video_filepath

            except:
                raise ValueError("An error occured. It may be that your input video is invalid.")
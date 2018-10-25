from faceai.Detection import *
import os
def main():
    execution_path = os.getcwd()
    model_path = ['MTCNN_model/PNet_landmark/PNet-30', 'MTCNN_model/RNet_landmark/RNet-22',
              'MTCNN_model/ONet_landmark/ONet-22']
    facedetector = VideoFaceDetection()
    facedetector.setModelTypeAsMTCNN()
    facedetector.setModelPath(model_path)
    facedetector.loadModel(min_face_size=24)
    inputpath = os.path.join(execution_path , "test.avi")
    outputpath  = os.path.join(execution_path , "test2new")
    re = facedetector.detectFacesFromVideo(input_file_path=inputpath, output_file_path=outputpath,frame_detection_interval=3)

if __name__ == '__main__':
    main()
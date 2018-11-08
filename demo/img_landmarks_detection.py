from faceai.Detection import *
from faceai.Alignment import LandmarksDetection
import os
import cv2

def main():
    execution_path = os.getcwd()
    model_path = ['MTCNN_model/PNet_landmark/PNet-30', 'MTCNN_model/RNet_landmark/RNet-22',
                  'MTCNN_model/ONet_landmark/ONet-22']
    input_path = os.path.join(execution_path, "land.jpg")
    output_path = os.path.join(execution_path, "imagenew.jpg")

    facedetector = FacesDetection()
    facedetector.setModelTypeAsMTCNN()
    facedetector.setModelPath(model_path)
    facedetector.loadModel(min_face_size=48)

    img,infs = facedetector.detectFacesFromImage(input_image=input_path,box_mark=False)
    dets = []
    for inf in infs:
        dets.append(inf["detection_details"])
    print('the number of faces: {:0>3d}'.format((len(infs))))

    landsdetector = LandmarksDetection()
    landsdetector.setModelTypeAsDAN()
    landsdetector.setModelPath("./Model/Model")
    landsdetector.loadModel()

    img,lands = landsdetector.detectLandmarksFromImage(img,dets,points_mark = True)
    cv2.imshow("t",img)

if __name__ == '__main__':
    main()
from faceai.Detection import *
from faceai.Alignment import LandmarksDetection
import os
import cv2
from skimage.io import imread, imsave
def main():
    execution_path = os.getcwd()
    input_path = os.path.join(execution_path, "3d.jpg")
    output_path = os.path.join(execution_path, "landnew.jpg")

    facedetector = FacesDetection()
    facedetector.setModelTypeAsMTCNN()
    facedetector.loadModel(min_face_size=48)

    img,infs = facedetector.detectFacesFromImage(input_image=input_path,box_mark=False)
    dets = []
    for inf in infs:
        dets.append(inf["detection_details"])
    print('the number of faces: {:0>3d}'.format((len(infs))))

    landsdetector = LandmarksDetection()
    landsdetector.setModelTypeAsDAN()
    #landsdetector.setModelTypeAsPRNet()
    landsdetector.loadModel()

    img,lands = landsdetector.detectLandmarksFromImage(img,dets,points_mark = True)
    cv2.imshow("t",img)
    cv2.imwrite("landnew.jpg", img)

if __name__ == '__main__':
    main()
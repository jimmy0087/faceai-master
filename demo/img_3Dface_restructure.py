from faceai.Detection import FacesDetection
from faceai.Alignment import LandmarksDetection
from faceai.ThrDFace import ThreeDimRestructure
import os
import cv2
from skimage.io import imread, imsave
import numpy as np
def main():
    execution_path = os.getcwd()
    input_path = os.path.join(execution_path, "3d_2.jpg")
    output_path = os.path.join(execution_path, "landnew.jpg")

    facedetector = FacesDetection()
    facedetector.setModelTypeAsMTCNN()
    facedetector.loadModel(detection_speed='fast',min_face_size=12)

    img,infs = facedetector.detectFacesFromImage(input_image=input_path,box_mark=False)
    dets = []
    for inf in infs:
        dets.append(inf["detection_details"])
    print('the number of faces: {:0>3d}'.format((len(infs))))

    landsdetector = ThreeDimRestructure()
    landsdetector.setModelTypeAsPRNet()
    landsdetector.loadModel()

    img_3d = landsdetector.restructure3DFaceFromImage(img,dets=dets,depth=True,pose=True)
    for img_ in img_3d:
        cv2.imshow("vertices",img_['img_show']['vertices'])
        cv2.imshow("pose", img_['img_show']['pose'])
        cv2.imshow("depth", img_['img_show']['depth'])



if __name__ == '__main__':
    main()
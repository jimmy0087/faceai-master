from faceai.Detection import *
import os
import cv2

def main():
    execution_path = os.getcwd()
    model_path = ['MTCNN_model/PNet_landmark/PNet-30', 'MTCNN_model/RNet_landmark/RNet-22',
                  'MTCNN_model/ONet_landmark/ONet-22']
    input_path = os.path.join(execution_path, "image.jpg")
    output_path = os.path.join(execution_path, "imagenew.jpg")
    # image_arr = cv2.imread(input_path)
    # image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    facedetector = FacesDetection()
    facedetector.setModelTypeAsMTCNN()
    facedetector.setModelPath(model_path)
    facedetector.loadModel(min_face_size=12)

    img,dets = facedetector.detectFacesFromImage(input_image=input_path,box_mark=True)
    #cv2.imshow("t",img)
    cv2.imwrite("imagenew.jpg", img)
    print('the number of faces: {:0>3d}'.format((len(dets))))

if __name__ == '__main__':
    main()
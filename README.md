# FaceAI

A python library built to realize face detection with Deep Learning using simple and few lines of code.

---

Now **FaceAI** is an original version and only support MTCCN to detect face.
Eventually, **FaceAI** will offer others various kind of applications about face.

### Update Record
- V0.0.1 : Initialize the file structure of FaceAI and add MTCCN model.

### Dependencies

- Tensorflow 1.10.0 (and later versions)

- Numpy 1.13.1 (and later versions)

- OpenCV

- Matplotlib

- easydict

- PIL

- scipy

### Demo

- Face Detection 
 **input image**
![image](https://raw.githubusercontent.com/jimmy0087/faceai-master/master/demo/image.jpg)
 **output image**
![image](https://raw.githubusercontent.com/jimmy0087/faceai-master/master/demo/imagenew.jpg)
```
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
    facedetector.loadModel(min_face_size=24)

    dets = facedetector.detectFacesFromImage(input_image=input_path, output_image_path=output_path,input_type="array")
    print('the number of faces: {:0>3d}'.format((len(dets))))

if __name__ == '__main__':
    main()
```



### Installation

To install ImageAI, run the python installation instruction below in the command line: 

pip install 

### References

- Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
[https://arxiv.org/abs/1604.02878v1](https://arxiv.org/abs/1604.02878v1)


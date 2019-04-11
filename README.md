# FaceAI

A python library built to realize face detection and face landmark with Deep Learning using simple and few lines of code.And my [website](https://jimmy0087.github.io/) will follow Latest paper and poster some my personal comment.

<img src="http://ww1.sinaimg.cn/thumbnail/0061KkpRly1fxtaimmmxtj308k08qq2w.jpg"/>

---

Now **FaceAI** is an original version and only support MTCCN to detect face.
Eventually, **FaceAI** will offer others various kind of applications about face.

### Update Record
- V0.3.0 : Add 3D face restruction module.
- V0.2.0 : Add download pre-trained automatically from google drive.
- V0.1.0 : Add landmarks detection module.
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

- ***Face Detection*** 
  **input image:**

  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fxunv27ajbj31kw0vyalc.jpg"/>
 
   **output image:**

  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fxunxmtoohj31kw0vytzw.jpg"/>

  **code:**
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

      img,dets = facedetector.detectFacesFromImage(input_image=input_path,box_mark=True)
      cv2.imwrite("imagenew.jpg", img)
      print('the number of faces: {:0>3d}'.format((len(dets))))

  if __name__ == '__main__':
      main()
  ```

- ***Face Landmarks Detection*** 
  **input image:**

  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fxuo6uxak3j30sg0hs75r.jpg"/>
 
  **output image:**

  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fxuo712hx4j30sg0hstbw.jpg"/>

  **code:**
  ```
  from faceai.Detection import *
  from faceai.Alignment import LandmarksDetection
  import os
  import cv2

  def main():
      execution_path = os.getcwd()
      model_path = ['MTCNN_model/PNet_landmark/PNet-30', 'MTCNN_model/RNet_landmark/RNet-22',
                  'MTCNN_model/ONet_landmark/ONet-22']
      input_path = os.path.join(execution_path, "land.jpg")
      output_path = os.path.join(execution_path, "landnew.jpg")

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
      cv2.imwrite("landnew.jpg", img)

  if __name__ == '__main__':
      main()
  ```

- ***3D Face Restruction*** 

  **input image:**

  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fy5dveivydj31hc0xcahj.jpg"/>
 
  **output image:**
    dense image
  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fy5e24oyj3j31hc0xck9l.jpg"/>
    pose image
  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fy5e1n0xhcj31hc0xcdv5.jpg"/>
    depth image
  <img src="http://ww1.sinaimg.cn/large/0061KkpRly1fy5gfq56jcj31hc0xcgmj.jpg"/>

  **code:**
  ```
  from faceai.Detection import FacesDetection
  from faceai.Alignment import LandmarksDetection
  from faceai.ThrDFace import ThreeDimRestructure
  import os
  import cv2

  def main():
    execution_path = os.getcwd()
    input_path = os.path.join(execution_path, "3d.jpg")
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
        cv2.imwrite("3d_vertices.jpg", img_['img_show']['vertices'])
        cv2.imwrite("3d_pose.jpg", img_['img_show']['pose'])
        cv2.imwrite("3d_depth.jpg", img_['img_show']['depth'])

  if __name__ == '__main__':
    main()
  ```
### Installation

To install ImageAI, run the python installation instruction below in the command line: 

pip install faceai-0.1.2-py2.py3-none-any.whl

### Pre-trained model

if you can not download model ,please download by hand,and code will remind you where you should uncompress these file.
> *mtcnn pre-trained model will download at C:/Users/Administrator/.faceai/mtcnn

- [DAN](https://drive.google.com/file/d/1Qh_OWZROneM01q8fxS9Clf7zfj5FBTdd/view?usp=sharing)

- [MTCNN](https://drive.google.com/file/d/1RRj9iPndDx3KXyjfG1cpfug-1X8RyDh5/view?usp=sharing)


### References

- [*Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks*](https://arxiv.org/abs/1604.02878v1)

- [*Deep Alignment Network: A convolutional neural network for robust face
alignment*](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Kowalski_Deep_Alignment_Network_CVPR_2017_paper.pdf)

- [*Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network*](https://arxiv.org/abs/1803.07835) [[code](https://github.com/YadiraF/PRNet)]
from faceai.Detection import FacesDetection
from faceai.Alignment import LandmarksDetection
from faceai.ThrDFace import ThreeDimRestructure
from faceai.Utils.visualization import show_3d_point,show_3d_mesh
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
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

    fig = plt.figure()
    show_vertice = []
    for img_ in img_3d:
        index = np.random.randint(0, high=len(img_['img_3d_inf']['vertices']), size=20000)
        show_vertice = img_['img_3d_inf']['vertices'][index]
        show_3d_point(np.array(show_vertice),img_['img_3d_inf']['color'][index])
        show_3d_point(img_['img_3d_inf']['landmarks_3d'])

        a = fig.add_subplot(2, 2, 1)
        a.axis('off')
        a.imshow(img[:, :, [2,1,0]])

        a = fig.add_subplot(2, 2, 2)
        a.axis('off')
        a.imshow(img_['img_show']['vertices'][:,:,[2,1,0]])

        a = fig.add_subplot(2, 2, 3)
        a.axis('off')
        a.imshow(img_['img_show']['pose'][:,:,[2,1,0]])

        a = fig.add_subplot(2, 2, 4)
        a.axis('off')
        a.imshow(img_['img_show']['depth'])
        plt.show()

        # cv2.imwrite("3d_vertices_mat.jpg", img_['img_show']['vertices'])
        # cv2.imwrite("3d_vertices.jpg", img_['img_show']['vertices'])
        # cv2.imwrite("3d_pose.jpg", img_['img_show']['pose'])
        # cv2.imwrite("3d_depth.jpg", img_['img_show']['depth'])


if __name__ == '__main__':
    main()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .colors import label_color


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def draw_landmarks(image, landmarks, color = label_color(0), thickness=2):
    """ Draws a landmarks on an image with a given color.

    # Arguments
        image     : The image to draw on.
        landmarks       : A list of (68,2) elements.
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(landmarks).astype(np.int32)
    for i in range(b.shape[0]):
        st = b[i]
        cv2.circle(image, (st[0], st[1]), thickness, color,-1)
        if i in end_list:
            continue
        ed = b[i + 1]
        cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)

def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, detections, color=None, generator=None):
    """ Draws detections in an image.

    # Arguments
        image      : The image to draw on.
        detections : A [N, 4 + num_classes] matrix (x1, y1, x2, y2, cls_1, cls_2, ...).
        color      : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        generator  : (optional) Generator which can map label to class name.
    """
    for d in detections:
        label   = np.argmax(d[4:])
        c       = color if color is not None else label_color(label)
        score   = d[4 + label]
        caption = (generator.label_to_name(label) if generator else str(label)) + ': {0:.2f}'.format(score)
        draw_caption(image, d, caption)

        draw_box(image, d, color=c)


def draw_annotations(image, annotations, color=(0, 255, 0), generator=None):
    """ Draws annotations in an image.

    # Arguments
        image       : The image to draw on.
        annotations : A [N, 5] matrix (x1, y1, x2, y2, label).
        color       : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        generator   : (optional) Generator which can map label to class name.
    """
    for a in annotations:
        label   = a[4]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(generator.label_to_name(label) if generator else label)
        draw_caption(image, a, caption)

        draw_box(image, a, color=c)

def show_3d_point(data):
    import matplotlib.colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(data)>1000:
        colormap = plt.get_cmap("rainbow")
    else:
        colormap = plt.get_cmap("winter")
    norm = matplotlib.colors.Normalize(vmin=min(data[:,2]), vmax=max(data[:,2]))
    ax.scatter(data[:,0], data[:,1], data[:,2], c=colormap(norm(data[:,2])) ,marker='.',alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.axis('off')
    plt.show()

def show_3d_mesh(data):
    from scipy.interpolate import griddata
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(data[:,0], data[:,1])
    Z = griddata(data[:,0:2],data[:,2],(X,Y),method='nearest')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

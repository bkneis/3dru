import cv2
import numpy as np
import dlib


def detect_face(img):
    """Detect a face in an image using cascade classifier

    :param img: Opencv Mat class representing a gray scale image
    :return x and y coords for the facial region
    """
    print('Detecting faces in the image')
    e1 = cv2.getTickCount()
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)

    if len(dets) > 1:
        print("Warning, more than 1 face detected")

    d = dets[0]
    rect = d.left(), d.top(), abs(d.left() - d.right()), abs(d.top() - d.bottom())
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print('Detecting the face took %.6f seconds' % time)
    return rect


def remove_background(img, face_coords):
    """Remove the background of the image using grab cut algorithm

    :param img: Opencv Mat class representing a gray scale image
    :param face_coords: x and y coords of the facial region
    :return: Opencv Mat representing the image without a background
    """
    print('Removing background from facial region')
    e1 = cv2.getTickCount()
    mask = np.zeros(img.shape[:3], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    rect = face_coords
    cv2.grabCut(img, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print('Removing the background took %.6f seconds' % time)
    return img * mask2[:, :, np.newaxis]

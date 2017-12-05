import dlib
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

predictor_path = 'trainingData/shape_predictor_68_face_landmarks.dat'

plot = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

f = 'data/face3.png'
print("Processing file: %s" % f)
img = cv2.imread(f)

mask = np.zeros(img.shape[:2], np.uint8)


def calc_distance(x, y, x2, y2):
    x_diff = abs(x - x2)
    y_diff = abs(y - y2)
    return math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))


def convert_detect_to_rect(d):
    return d.left(), d.top(), abs(d.left() - d.right()), abs(d.top() - d.bottom())


# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: %i" % len(dets))

if len(dets) > 1:
    print("Warning, more than 1 face detected")

d = dets[0]
# cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)
shape = predictor(img, d)
rect = (d.left(), d.top(), d.right(), d.bottom())
print(rect)
for idx in range(1, shape.num_parts):
    point = shape.part(idx)
    point2 = shape.part(idx - 1)
    distance = calc_distance(point.x, point.y, point2.x, point2.y)
    if distance < 100:
        cv2.line(img, (point.x, point.y), (point2.x, point2.y), (255, 255, 255), 5)
print(shape.num_parts)
# mask = mask[d.top():d.bottom(), d.left():d.right()]
cv2.imwrite('results/faceLandmarks.png', img)

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
rect = (479, 911, 555, 555)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

if plot:
    plt.imshow(img)
    plt.colorbar()
    plt.show()
else:
    cv2.imwrite("results/croppedFace.png", img)

# newmask is the mask image I manually labelled
newmask = cv2.imread('results/faceLandmarks.png', 0)
# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img*mask[:, :, np.newaxis]
cv2.imwrite('results/croppedFace2.png', img)

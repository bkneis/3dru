import dlib
import cv2

predictor_path = '/home/arthur/Downloads/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

f = 'data/face1.png'
print("Processing file: {}".format(f))
img = cv2.imread(f)

#win.clear_overlay()
#win.set_image(img)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for k, d in enumerate(dets):
    # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #     k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.
    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)
    shape = predictor(img, d)

    for idx in range(0, shape.num_parts):
        if idx == 0:
            continue
        print(shape.part(idx))
        point = shape.part(idx)
        point2 = shape.part(idx - 1)
        cv2.line(img, (point.x, point.y), (point2.x, point2.y), (255, 0, 0), 5)

#win.add_overlay(dets)
cv2.imwrite('results/faceLandmarks.png', img)
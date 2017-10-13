import cv2

face_cascade = cv2.CascadeClassifier('sift/trainingData/haarcascade_frontalface_default.xml')


def cropface(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    x, y, w, h = faces[0]

    img = img[y:(y + h), x:(x + w)]

    if len(faces) < 1:
        print("Potentially unexpected behaviour, no face was detected")
    elif len(faces) > 1:
        print("Potentially unexpected behaviour, more than 1 face detected")

    return img

import cv2


sift = cv2.xfeatures2d.SIFT_create()


def match(descriptors, descriptors2):
    """Match the SIFT features using FLANN

    :param descriptors: descriptors of the 3dru features extracted from image 1
    :param descriptors2: descriptors of the 3dru features extracted from image 2
    :return: List of matched descriptors
    """
    print('Attempting to match features from the 2 images')
    e1 = cv2.getTickCount()
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors, descriptors2, k=2)
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print('Matching the features took %.6f seconds' % time)
    return matches


def get_features(img):
    """Retrieve the features and their descriptors using 3dru

    :param img:
    :return:
    """
    print('Detecting and computing features in image using 3dru')
    e1 = cv2.getTickCount()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print('Extracting the features took %.6f seconds' % time)
    return keypoints, descriptors


def draw_matches(img, img2, keypoints, keypoints2, matches):
    """Draw the matched features of the 2 images and return an image with them illustrated

    :param img:
    :param img2:
    :param keypoints:
    :param keypoints2:
    :param matches:
    :return: Mat: image with drawn matches
    """
    print('Drawing matches of the 2 images on a new image')
    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matches_mask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matches_mask,
                       flags=0)

    return cv2.drawMatchesKnn(img, keypoints, img2, keypoints2, matches, None, **draw_params)

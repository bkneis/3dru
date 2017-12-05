import cv2


def generate_disparity_maps(lengths, baselines):
    """Generate disparity maps for a set of images

    Given a folder containing a set of folders that contain each a set of stereo images,
    calculate the disparity maps for each image pair using varying parameters of number
    of disparities and block size for the block matching algorithm

    The purpose of this function is to illustrate the effects of varying parameters
    for the block matching algorithm used when corresponding images
    """
    for length in lengths:
        for base in baselines:
            imgL = cv2.imread('/home/arthur/Desktop/%il_%ib/left.png' % (length, base), 0)
            imgR = cv2.imread('/home/arthur/Desktop/%il_%ib/right.png' % (length, base), 0)

            output_path = '/home/arthur/disparities/%il_%ib' % (length, base)

            d = 16
            for j in range(0, 10):
                b = 5
                for i in range(0, 10):
                    stereo = cv2.StereoBM_create(numDisparities=d, blockSize=b)
                    disparity = stereo.compute(imgL, imgR)
                    cv2.imwrite('%s/img%i_disparity=%i_blocksize=%i.png' % (output_path, i, d, b), disparity)
                    b += 2
                d += 16


if __name__ == '__main__':
    generate_disparity_maps([30, 50], [40, 50, 60])

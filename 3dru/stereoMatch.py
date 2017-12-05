# Python 2/3 compatibility
from __future__ import print_function

import cv2
import sys
import numpy as np
import preprocessor as pre
import subprocess


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    """Extract features from 2 images using 3dru and match them using FLANN and save matches to an image"""

    print('Reading images')
    img = cv2.imread('data/face1.png')
    img2 = cv2.imread('data/face2.png')
    if img is None or img2 is None:
        print('Error: Could not read one of the images')
        return 0

    face_coords = pre.detect_face(img)
    x, y, w, h = face_coords
    x -= 300
    y -= 300
    w += 600
    h += 600

    imgL = img[y:(y + h), x: (x + w)]
    imgR = img2[y:(y + h), x: (x + w)]

    imgL = cv2.pyrDown(imgL)  # downscale images for faster processing results/face3bg.png
    imgR = cv2.pyrDown(imgR)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...', )
    h, w = imgL.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    write_ply('output/%s' % sys.argv[1], out_points, out_colors)
    print('%s saved' % 'out.ply')

    print('Converting ply to pcd')
    print('output/%s output/%s' % (sys.argv[1], sys.argv[2]))

    subprocess.call('/usr/local/pcl/build/bin/pcl_converter output/%s output/%s' % (sys.argv[1], sys.argv[2]), shell=True)

    print('Done.')


if __name__ == "__main__":
    main()

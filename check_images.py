import argparse
import glob
import yaml
from shutil import copy2
import cv2
import numpy as np
import os
'''
This script checks each image in the input folder to test whether (a) the chessboard can be found in the image and (b) no cv2 error is thrown.
'''

parser = argparse.ArgumentParser()

parser.add_argument('--images', type=str)
parser.add_argument('--out', type=str, default='sorted_files')

args = parser.parse_args()
image_files = glob.glob(f'{args.images}/*.jpg')
image_files.sort()

CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

outputs = {
    'no_chessboard' : [],
    'cv_err' : [],
    'good_images' : []
}

for fname in image_files:
    print(f"Processing file {fname}.")
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    print(f"Looking for the chess board...")
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    print(f"Did we find the chessboard? {ret}")
    if ret == True:
        objpoints = [objp]
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints=[corners]
    else:
        outputs['no_chessboard'].append(fname)
        continue

    N_OK = 1
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    try:
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        outputs['good_images'].append(fname)
    except Exception as e:
        print(f"got the cv exception: {e}")
        outputs['cv_err'].append(fname)
    
    


with open(os.path.join(args.out,'meta.yaml'), 'w') as f:
    yaml.dump(outputs, f)
print("copying the files")
# don't modify filesystem until everything else has completed
for k,v in outputs.items():
    if not os.path.exists(os.path.join(args.out,k)):
        os.makedirs(os.path.join(args.out,k))
    for fpath in v:
        fname = os.path.basename(fpath)
        copy2(fpath, os.path.join(args.out, k, fname))


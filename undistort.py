import numpy as np
import cv2
import sys
import glob
import argparse
import os
from tqdm import tqdm
# You should replace these 3 lines with the output in calibration step
# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

#cs mount
# DIM=(4056, 3040)
# K=np.array([[994.0491974872951, 0.0, 2062.33259320713], [0.0, 994.109826917038, 1506.8248039837533], [0.0, 0.0, 1.0]])
# D=np.array([[-0.0059077571932170565], [-0.004934877307889183], [0.0007128571795357302], [-0.0003785613682513791]])


#m12
# DIM=(4056, 3040)
# K=np.array([[987.5291174568782, 0.0, 2024.6472735612485], [0.0, 987.7859990883527, 1524.754759211665], [0.0, 0.0, 1.0]])
# D=np.array([[-0.018655061411515527], [0.007779105264154045], [-0.0045057681472640025], [0.0003867090406766997]])
# DIM=(4056, 3040)
# K=np.array([[978.7299346438658, 0.0, 2030.0117631492542], [0.0, 977.5772451671866, 1525.45802537491], [0.0, 0.0, 1.0]])
# D=np.array([[-0.010049094493025242], [-0.0020168893214384314], [0.0004935384645605944], [-0.00046938729406952837]])

# DIM=(4032, 3040)
# K=np.array([[988.1138204372988, 0.0, 2024.1319076494988], [0.0, 988.4489533032867, 1521.4834690780724], [0.0, 0.0, 1.0]])
# D=np.array([[-0.013673858142318831], [-0.00230694568968787], [0.001815386243123689], [-0.0008660795105125041]])

# DIM=(4032, 3040)
# K=np.array([[1322.4055727913471, 0.0, 2033.0346323930382], [0.0, 1321.4448263081883, 1518.0594881239615], [0.0, 0.0, 1.0]])
# D=np.array([[-0.26970294235177733], [0.13219272208967142], [-0.04395805249003419], [0.006889521495453855]])

# Before 9/6
# DIM=(4032, 3040)
# K=np.array([[1016.5515347866365, 0.0, 2028.6228893509635], [0.0, 1016.1122888494706, 1513.6108385639468], [0.0, 0.0, 1.0]])
# D=np.array([[-0.04153696712152178], [0.012263123674721975], [-0.0032037098308749725], [-0.00014979494407106234]])

# 9/6 fhd
# DIM = (1920, 1200)
# K=np.array([[751.0029981007824, 0.0, 966.7153295996677], [0.0, 752.5350158258151, 634.5683305375871], [0.0, 0.0, 1.0]])
# D=np.array([[-0.5342650764868064], [0.6408770768896442], [-0.7086135063056186], [0.4059440700351538]])

# DIM=(1920, 1200)
# K=np.array([[755.7302234414252, 0.0, 966.436550411427], [0.0, 757.2885588757678, 635.0059308840977], [0.0, 0.0, 1.0]])
# D=np.array([[-0.5433444517433353], [0.6189897841054645], [-0.6267761594514107], [0.34665314030141525]])

DIM=(1920, 1200)
K=np.array([[842.8051737345792, 0.0, 969.2857552509467], [0.0, 845.2443143454481, 634.6076622950868], [0.0, 0.0, 1.0]])
D=np.array([[-0.7560057655709489], [1.1471803047215792], [-1.4634661133229243], [0.8978357196785213]])

def undistort(img_path):
    '''Undistort image and save it to disk'''
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)


    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    output_path = os.path.join(os.path.dirname(img_path), 'undistorted', os.path.basename(img_path))
    # cv2.imshow(f"undistorted {img_path}", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(output_path, undistorted_img)

    return output_path, undistorted_img

def undistort_img(img):
    '''Undistort image and save it to disk'''
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def undistort2(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    print(dim1)
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)


    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    output_path = os.path.join(os.path.dirname(img_path), 'undistorted', os.path.basename(img_path))
    cv2.imwrite(output_path, undistorted_img)
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Undistort images')
    parser.add_argument('--dir', type=str, default='', help='Path to images')
    args = parser.parse_args()

    for i,fname in tqdm(enumerate(glob.glob(str(os.path.join(args.dir, '*.jpg'))))):
        # undistort2(fname, balance=1.00)
        undistort(fname)

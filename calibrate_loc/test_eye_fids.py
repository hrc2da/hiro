from pupil_apriltags import Detector
import argparse
import glob
import os
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='Detect apriltags and save the detection in a file')
parser.add_argument('-i', '--input', help='input folder', default='undistorted')

args = parser.parse_args()

at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)

imgs = glob.glob(args.input+'/*.jpg')
imgs= ["/home/pi/Pictures/ov1.png"]
if not os.path.exists(args.input+'/detected'):
    os.makedirs(args.input+'/detected')

for img_f in imgs:
    img = cv2.imread(img_f,cv2.IMREAD_GRAYSCALE)
    tags = at_detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
    print(f"found {len(tags)} tags")
    for tag in tags:
        pts = tag.corners.astype(int)
        pts = np.append(pts, [pts[0]], axis=0)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], False, (0,0,0), 2)
    print(f'Writing image to {os.path.join(os.path.dirname(img_f),"detected",f"{os.path.basename(img_f)}_detected.jpg")}')
    cv2.imwrite(os.path.join(os.path.dirname(img_f),'detected',f'{os.path.basename(img_f)}_detected.jpg'), img)


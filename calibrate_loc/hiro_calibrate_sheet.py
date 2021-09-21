from hiro_lightweight_interface import HIROLightweight
import argparse
from pupil_apriltags import Detector
import pdb
import cv2
import numpy as np
import csv
import os
import atexit
from static_localization import localize_notecard

parser = argparse.ArgumentParser()
parser.add_argument("--startx")
parser.add_argument("--starty")
parser.add_argument("--outfile", default="sheet.csv")
parser.add_argument("--stride", default=20)
parser.add_argument("--sheet_stride", default=5)

at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)

n_cols = 10
n_rows = 5

def generate_target_locs(n_cols, n_rows, startx, starty, stride, offset):
    l = 22.5
    k = 12.8
    startx += offset
    starty += offset
    target_locs = {}
    for i in range(n_rows):
        for j in range(n_cols):
            fid = i*n_cols + j
            x = startx + j*stride - l
            y = starty + i*stride - k
            target_locs[fid] = [(x,y),(x,y+n_rows*stride)] # the pattern repeats twice
    return target_locs




def read_sheet(view, pose, pixel2mm, target_locs):
    tags = at_detector.detect(view, estimate_tag_pose=False, camera_params=None, tag_size=None)
    detected_locs = {}
    for fid in target_locs.keys():
        detected_locs[fid] = [[],[]]
    for tag in tags:
        fid = tag.tag_id
        target1, target2 = target_locs[fid]
        card_loc = localize_notecard(tag, float(pixel2mm), np.array(pose))
        # let's assume that, since the fids are something like 100 mm apart, so the one closer in y is the relevant target
        ydist1 = np.abs(card_loc[1]-target1[1])
        ydist2 = np.abs(card_loc[1]-target2[1])
        if ydist1 < ydist2:
            # target1 is the target
            detected_locs[fid][0] = card_loc[:2]
        else:
            detected_locs[fid][1] = card_loc[:2]
    return detected_locs


    # if orientation == 'horizontal':

        # n_cols = 13
        # n_rows = 9
        # columns = [[] for c in range(n_cols)]
        # for tag in tags:
        #     fid = tag.tag_id
        #     card_loc = localize_notecard(tag, float(pixel2mm), np.array(pose))
        #     columns[int(fid)].append(card_loc)
        # for col in columns:
        #     col.sort(key=lambda x: x[1]) # sort each column by y ascending
        # first_detections = [col[0] for col in columns]
        
                
        # cols = [[t for t in tags if t.tag_id==i] for i in range(n_cols)]
        # for c in cols:
        #     c.sort(key=lambda x: x.center[1])
        # centers = [[t.center for t in col] for col in cols]
        # xs = [[c[0] for c in col] for col in centers]
        # ys = [[c[1] for c in col] for col in centers]

    pdb.set_trace()

if __name__=='__main__':
    args = parser.parse_args()
    startx = int(args.startx)
    starty = int(args.starty)
    stride = int(args.stride)
    sheet_stride = int(args.sheet_stride)
    offset = stride
    h = HIROLightweight()
    h.move(np.array([[0],[200],[240]]))
    view = h.capture(imagepath="calibrate.jpg")
    view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    pose = h.position
    pixel2mm = float(pose[2,0]*h.pixel_scalar + h.pixel_intercept)

    write_headers = True
    if os.path.exists(args.outfile):
        write_headers = False
    outfile = open(args.outfile, 'a+')
    writer = csv.writer(outfile)
    if write_headers:
        writer.writerow(('fid','x','y','xa','ya'))
    internal_counter = 0
    while True:
        print(f"Place sheet at {startx}, {starty}")
        input("Press Enter to capture")
        view = h.capture(imagepath="calibrate.jpg")
        view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        target_locs = generate_target_locs(n_cols, n_rows, startx, starty, stride, offset)
        detected_locs = read_sheet(view, pose, pixel2mm, target_locs)
        for fid,targets in target_locs.items():
            target1,target2 = targets
            detected1,detected2 = detected_locs[fid]
            if len(detected1) > 0:
                writer.writerow([fid,*target1,*detected1])
                outfile.flush()
            if len(detected2) > 0:
                writer.writerow([fid,*target2,*detected2])
                outfile.flush()
        # skip to avoid overlap
        if internal_counter == 15:
            startx += 185
            internal_counter = 0
        else:
            startx += sheet_stride
            internal_counter += 5

    outfile.close()
    hiro.shutdown()
    

import pyuarm
import csv
import getch
import numpy as np
import time
import os
import sys
sys.path.append('../')
from hiro_lightweight_interface import HIROLightweight

'''
Procedure:
1) generate a point
2) user moves the card to that point
3) store the point
'''

def wait_for_card(target, hiro):
    ''' Print prompt. Wait for user to place the card. Return on "enter"'''
    print(f"Place card at {target}")
    input("Press enter to record the location")
    while True:
        fmap = hiro.get_fiducial_map(raw=True)
        if len(list(fmap.keys())) < 1:
            try_again = input("Didn't find the card. Press enter to try again or q to skip.")
            if try_again == 'q':
                return None
            else:
                continue
        elif len(list(fmap.keys())) > 1:
            try_again = input("Found more than one card. Press enter to try again or q to skip.")
            if try_again == 'q':
                return None
            else:
                continue
        else:
            # found one card, return the location
            hiro.beep(1)
            return list(fmap.values())[0]
        

if __name__=='__main__':
    hiro = HIROLightweight()
    hiro.move(np.array([[0],[200],[200]]))
    filename = 'cam_locs.csv'
    data = []
    times = []
    write_headers = True
    if os.path.exists(filename):
        write_headers = False
    outfile = open(filename, 'a+')
    writer = csv.writer(outfile)
    if write_headers:
        writer.writerow(('x','y','xa','ya'))
    xmin = -90
    xmax = 100 #inclusive
    ymin = 120
    ymax = 200 # inclusive
    for i in range(xmin,xmax+1,10):
        start = time.time()
        for j in range(ymin,ymax+1,5):
            target = [i,j,60]
            observed_pos = wait_for_card(target, hiro)
            if observed_pos is not None:
                print(f"Writing {observed_pos[:2]} for {target[:2]}...")
                data.append((*target[:2],*observed_pos[:2]))
                writer.writerow(data[-1])
                outfile.flush()
            else:
                print(f"Skipping point {target}")
        col_time = time.time()-start
        print(f"col_time:{col_time} s")
        times.append(col_time)
    print(times)
    outfile.close()
    hiro.shutdown()
    


     
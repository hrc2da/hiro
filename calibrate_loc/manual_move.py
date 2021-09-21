import pyuarm
import csv
import getch
import time
import os
from copy import deepcopy

'''
Procedure:
1) generate a point
2) user moves the robot to that point
3) store the point
'''

def handle_user_move(arm, target, home=False, step=2.5):
    '''Wait for user to move the robot. Return on "q"'''
    pose = deepcopy(target)
    
    home_pos = (0,200,200)
    if home:
        arm.set_position(*home_pos, wait=True)
    
    while True:
        print(f"Target pose: {target}")
        arm.set_position(*pose, wait=True)
        print(f"Current pose: {pose}")
        print("Enter a move or r to save or q to quit")
        command = getch.getch()
        if command == 'r':
            # finalize pose
            return pose
        if command == 'q':
            # give up
            return None
        if command == 't':
            # test the current pos
            arm.set_position(*home_pos, wait=True)
            time.sleep(1)
        elif command == 'w':
            pose[1] -= step
        elif command == 's':
            pose[1] += step    
        elif command == 'a':
            pose[0] += step
        elif command == 'd':
            pose[0] -= step
        elif command == 'k':
            pose[2] -= step
        elif command == 'j':
            pose[2] += step

if __name__=='__main__':
    arm = pyuarm.UArm()
    arm.connect()
    home_pos = (0,200,200)
    arm.set_position(*home_pos, wait=True)
    data = []
    filename = 'move_locs.csv'
    write_headers = True
    if os.path.exists(filename):
        write_headers = False
    outfile = open(filename, 'a+')
    writer = csv.writer(outfile)
    if write_headers:
        writer.writerow(('x','y','z','xa','ya','za'))
    xmin = -325
    xmax = -200 #inclusive
    ymin = 50
    ymax = 100 # inclusive
    for i in range(xmin,xmax+1,25):
        for j in range(ymin,ymax+1,25):
            target = [i,j,80]
            observed_pos = handle_user_move(arm, target, True)            
            if observed_pos is not None:
                print(f"Writing {observed_pos} for {target}...")
                data.append((*target,*observed_pos))
                writer.writerow(data[-1])
                outfile.flush()
            else:
                print(f"Skipping point {target}")
    outfile.close()
    arm.set_position(*home_pos, wait=True)


     
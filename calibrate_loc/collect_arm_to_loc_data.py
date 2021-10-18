# print a coordinate
# wait for user to move to point and hit enter 
# spin up a cv2 video capture stream
# look for a fiducial
# capture the center
# turn the servos on
# move to (0,200,200)
# move to target @ 50 (to be safe)
# look for a fiducial
# capture the center
# save {"target": ("measured", "moved")}

from numpy.core.arrayprint import DatetimeFormat
from pyuarm import UArm
from pupil_apriltags import Detector
import cv2
import atexit
import yaml
import time
data = []
filename = "move_calibration_data.yml"

def write_data():
    global data
    global filename
    with open(filename, 'w+') as outfile:
        yaml.dump(data,outfile)
atexit.register(write_data)
def find_fid(detector):
    while True:
        capture = cv2.VideoCapture(1)
        _,img = capture.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('temp.jpg', img)
        capture.release()
        tags = detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
        if len(tags) < 1:
            again = input(f"Couldn't find fidicual. Enter y to try again")
            if again != "y":
                print("skipping")
                return False
        else:
            return tags[0]




def main():
    at_detector = Detector(families='tag36h11',nthreads=1,quad_decimate=1.0,quad_sigma=0.0,refine_edges=1,decode_sharpening=0.25,debug=0)

    arm = UArm()
    arm.connect()
    time.sleep(2)
    
    for i in range(-300,-200,50):
        for j in range(0,250,50):
            z_height=70
            print(f"Tring to go to ({i},{j},{z_height}).")
            arm.set_servo_attach(wait=True)
            arm.set_position(0,200,200,wait=True)
            arm.set_position(i,j,z_height,wait=True)
            success = input("Enter y if succesful")
            if success != 'y':
                print(f"Skipping point")
                continue
            #drop the height until it reaches the ground
            while True:
                stop = input("enter y when the tip is touching the ground")
                if stop == 'y':
                    break
                else:
                    z_height -= 5
                    arm.set_position(i,j,z_height, wait=True)

            
            measured_move = (i,j,z_height)

            # now turn off the servos and ask the person to corect the move
            input("Dropping arm, hit enter when ready.")
            arm.set_servo_detach()
            input("Move arm to correct location. Hit enter when done.")
            
            target_introspective = arm.get_position()
            data.append((measured_move, target_introspective))
            write_data()


if __name__=='__main__':
    main()
    write_data()





            


'''
Just puts cards into clusters blindly
mimicks what actual opperation may look like
tests pick and place fucntionality and cluster configuration
'''

from hiro_interface import *
import sys
from copy import copy
from nlp_utils import NoteParser
import string
import threading

hiro = HIRO(mute=False)
temp_photo_path = '/home/pi/hiro/views/view.jpg'
vid_done = True
vid_id = 0
def record_video(nframes=10,id=None):
    global vid_done
    vid_done = False
    camera = cv2.VideoCapture(1)
    frames = []
    for i in range(nframes):
        s,frame = camera.read()
        frames.append(frame)
    print("SAVINGG VIDEO")
    np.save(f'/home/pi/hiro/reading_imgs/positioned_by_hiro/vid_{id}',frames)
    camera.release()
    vid_done = True

seen = []
try:
    for i in range(500):
        # if vid_done:
        #     print("STARTING VIDEO")
        #     vid_thread = threading.Thread(target=record_video, args=(1000,vid_id))
        #     vid_thread.start()
        #     vid_id += 1
        # wait for card (blocking)
        new_id = hiro.find_new_card(seen,reposition=True)
        hiro.capture('/home/pi/hiro/reading_imgs/positioned_by_hiro/%d.jpg' % int(i+88))
        time.sleep(2)
        for i in range(5):
            hiro.camera.grab()
        
        # seen.append(new_id)
except:
    hiro.shutdown()
    
hiro.move(np.array([[0], [200], [200]]))

hiro.disconnect()



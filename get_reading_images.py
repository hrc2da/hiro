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

hiro = HIRO(mute=True)
temp_photo_path = '/home/pi/hiro/views/view.jpg'

seen = []

for i in range(500):
    # wait for card (blocking)
    new_id = hiro.find_new_card(seen,reposition=True)
    hiro.capture('/home/pi/Desktop/reading_imgs/positioned_by_hiro/%d.jpg' % int(i+45))
    seen.append(new_id)

hiro.move(np.array([[0], [200], [200]]))

hiro.disconnect()



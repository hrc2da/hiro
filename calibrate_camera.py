'''
script that takes pictures at different heights to calculate
the conversion between pixels and mm wrt the height

'''

from hiro_interface import *

hiro = HIRO()

heights = [100, 150, 200, 250]

for z in heights:
    hiro.move(np.array([[0], [200], [z]]))
    hiro.capture('/home/pi/Desktop/calibration/%d.jpg' % z)

hiro.disconnect()
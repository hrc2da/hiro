'''
script that takes pictures at different heights to calculate
the conversion between pixels and mm wrt the height

'''

from hiro_interface import *

hiro = HIRO()

heights = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]

for z in heights:
    hiro.move(np.array([[0], [200], [z]]))
    #print(z)
    hiro.capture('/home/pi/Desktop/calibration/%d.jpg' % z)

hiro.disconnect()
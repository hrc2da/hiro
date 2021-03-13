from hiro_interface import *

hiro = HIRO(mute=True)

#pos = np.array([[0],[290],[110]]) # spot to read card
pos = np.array([[0],[280],[200]]) # spot to wait at for new card

hiro.move(pos)


for i in range(100):
    hiro.capture('/home/pi/Desktop/calibration/%d.jpg' % i)
    input('press enter to continue')
hiro.disconnect()



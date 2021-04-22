from hiro_interface import *

hiro = HIRO(mute=True)

sweep_result = hiro.sweep(sweep_points = [(0,200)], sweep_height=240)

print(sweep_result)

hiro.disconnect()



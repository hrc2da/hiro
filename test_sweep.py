from hiro_interface import *

hiro = HIRO(mute=True)

sweep_result = hiro.sweep(sweep_points = [(-20,200),(-100,200),(-150,100),(-165,178)], sweep_height=220)

print(sweep_result)

hiro.disconnect()



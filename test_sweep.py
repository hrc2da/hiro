from hiro_interface import *

hiro = HIRO(mute=True)

sweep_result = hiro.sweep()

hiro.pick_place(sweep_result[4], ())

print(sweep_result)

hiro.disconnect()



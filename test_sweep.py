from hiro_interface import *

hiro = HIRO(mute=True)

sweep_result = hiro.sweep()

for loc in sweep_result.values():
    hiro.pick_place(loc, (0,280))


hiro.disconnect()



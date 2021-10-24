from hiro_lightweight_interface import HIROLightweight

interface = HIROLightweight()
dummy_spot = [-200,75,0]
move_test = [[[-50, 50, 0], [-100, 50, 0]], [[-100, 50, 0], [-50, 50, 0]], [dummy_spot, [-75, 75, 0]]]

print(interface.sort_moves(move_test, dummy_spot))



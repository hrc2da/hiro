from hiro_lightweight_interface import HIROLightweight
import requests
from config import Config
import numpy as np

interface = HIROLightweight()
dummy_spot = [-200,75,0]
move_test = [[1, [-50, 50, 0], [-100, 50, 0]], [2, [-100, 50, 0], [-50, 50, 0]], [3, dummy_spot, [-75, 75, 0]]]
MOVE_THRESHOLD = 15
ERROR_PNP_LOC = 1
ERROR_PNP_FALSE = 2
ERROR_MOVE_SORT = 3
print(interface.sort_moves(move_test, dummy_spot))

fmap = interface.get_fiducial_map(mask=None)
cur_cards = list(fmap.keys())
new_card = cur_cards[0]
cur_cards.sort()
cur_notes = [int(fid) for fid in cur_cards]
cur_locs = [fmap[fid][:2] for fid in cur_cards]
r = requests.post(Config.API_URL+'/addnote', json={"encoding_type":"rgb", "new_note": new_card, "notes": cur_notes, "locs": cur_locs, "operations": ["split"]})
res = r.json()
print(res)

new_loc_dict = {res['notes'][i]:res['locs'][i]+[0] for i in range(len(res['notes']))}
adds,moves,removes,invalids = interface.getmoves(new_loc_dict, fmap, add_fid=str(new_card), dummy_spot=[-200,75,0], threshold = MOVE_THRESHOLD)
print(f"adds:{adds}")
print(f"moves:{moves}")
print(f"removes:{removes}")
print(f"invalids:{invalids}")


n_tries = 3
cur_moves = moves
invalid_set = set()
while cur_moves != [] and n_tries > 0:
    for fid,start,stop in cur_moves:
        interface.move(np.array([[0],[200],[200]]))
        move_result = interface.pick_place(start, stop)
        if move_result == False:
            with open("errors.txt", "a") as f:
                f.write(str(ERROR_PNP_FALSE) + "\n")
            invalid_set.add((tuple(start), tuple(stop)))

    check_fiducials = interface.get_fiducial_map(mask=None)
    _, cur_moves, _, _ = interface.getmoves(new_loc_dict, check_fiducials, add_fid=new_card, dummy_spot=[-200,75,0], threshold = MOVE_THRESHOLD * 3)
    if cur_moves == [-1]:
        with open("errors.txt", "a") as f:
            f.write(str(ERROR_MOVE_SORT) + "\n")
        cur_moves = []
    else:
        cur_moves = [i for i in cur_moves if (tuple(i[1]), tuple(i[2])) not in invalid_set]
        print(cur_moves)
        if cur_moves != []:
            with open("errors.txt", "a") as f:
                f.write(str(ERROR_PNP_LOC) + "\n")
    n_tries -= 1

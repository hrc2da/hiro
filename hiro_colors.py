from hiro_lightweight_interface import HIROLightweight
from config import Config
import requests
import cv2
import yaml
import numpy as np

# instantiate a HIRO object
hiro = HIROLightweight()
#add_zone = [(-300,0),(-200,100)]
add_zone = [(-335, 75), (-260,125)] # gonzalo - more accurate add_zone
MOVE_THRESHOLD = 15
FOCUS_THRESHOLD = 5
ERROR_PNP_LOC = 1
ERROR_PNP_FALSE = 2
ERROR_MOVE_SORT = 3

while True:
    print("looping")
    # spin until you detect a fiducial in the "new card zone"
    result = hiro.wait_for_card(loading_zone=add_zone, focus_threshold = FOCUS_THRESHOLD)
    new_card, loc, fmap = result
    # get the notes for the new card and map and POST the new word to the api to update the diagram
    note = int(new_card)
    cur_cards = list(fmap.keys())
    cur_cards.sort()
    cur_notes = [int(fid) for fid in cur_cards]
    cur_locs = [fmap[fid][:2] for fid in cur_cards]

    r = requests.post(Config.API_URL+'/addnote', json={"encoding_type":"rgb" ,"new_note": note, "notes": cur_notes, "locs": cur_locs, "operations": ["add","split"]})
    res = r.json()
    # + [0] is for rotation
    # new_loc_dict = {res['notes'][i]:res['locs'][i]+[0] for i in range(len(res['notes']))}
    # adds,moves,removes,invalids = hiro.getmoves(new_loc_dict, fmap, add_fid=new_card, dummy_spot=[-200,75,0])
    # print(f"adds:{adds}")
    # print(f"moves:{moves}")
    # print(f"removes:{removes}")
    # print(f"invalids:{invalids}")
    # for fid,start,stop in moves:
    #     hiro.move(np.array([[0],[200],[200]]))
    #     hiro.pick_place(start, stop)
    # assert len(adds) < 2 # assuming we can only add from one location
    # if len(adds) == 1:
    #     fid,stop = adds[0]
    #     hiro.pick_place(loc,stop)
    # print(r.json())

    new_loc_dict = {res['notes'][i]:res['locs'][i]+[0] for i in range(len(res['notes']))}
    adds,moves,removes,invalids = hiro.getmoves(new_loc_dict, fmap, add_fid=new_card, dummy_spot=[-200,75,0], MOVE_THRESHOLD)
    if moves == [-1]:
        with open("errors.txt", "a") as f:
            f.write(str(ERROR_MOVE_SORT) + "\n")
        moves = []
    print(f"adds:{adds}")
    print(f"moves:{moves}")
    print(f"removes:{removes}")
    print(f"invalids:{invalids}")

    n_tries = 3
    cur_moves = moves
    invalid_set = set()
    while cur_moves != [] and n_tries > 0:
        for fid,start,stop in cur_moves:
            hiro.move(np.array([[0],[200],[200]]))
            move_result = hiro.pick_place(start, stop)
            if move_result == False:
                with open("errors.txt", "a") as f:
                    f.write(str(ERROR_PNP_FALSE) + "\n")
                invalid_set.add((start, stop))
            
        check_fiducials = hiro.get_fiducial_map(mask=zone_mask)
        _, cur_moves, _, _ = hiro.getmoves(new_loc_dict, check_fiducials, add_fid=new_card, dummy_spot=[-200,75,0], MOVE_THRESHOLD * 3)
        if cur_moves == [-1]:
            with open("errors.txt", "a") as f:
                f.write(str(ERROR_MOVE_SORT) + "\n")
            cur_moves = []
        else:
            cur_moves = [i for i in cur_moves if (i[1], i[2]) not in invalid_set]
            if cur_moves != []:
                with open("errors.txt", "a") as f:
                    f.write(str(ERROR_PNP_LOC) + "\n")
        n_tries -= 1
        
    assert len(adds) < 2 # assuming we can only add from one location
    if len(adds) == 1:
        n_tries = 3
        success = False
        while not success and n_tries > 0:
            fid,stop = adds[0]
            move_result = hiro.pick_place(loc,stop)
            check_fiducials = hiro.get_fiducial_map(mask=zone_mask)
            if check_fiducials[fid] != stop:
                with open("errors.txt", "a") as f:
                    f.write(str(ERROR_PNP_LOC) + "\n")
                n_tries -= 1
            else:
                success = True
    
hiro.shutdown()

from hiro_lightweight_interface import HIROLightweight
from config import Config
import requests
import cv2
import yaml
import numpy as np
import datetime
import json
with open('fiducial_dict.yaml', 'r') as infile:
    fiducial_dict = yaml.load(infile)

notes_dict ={v:k for k,v in fiducial_dict.items()}
# instantiate a HIRO object
hiro = HIROLightweight()
add_zone = [(-300,-50),(-200,100)]

MOVE_THRESHOLD = 15
CHANGE_THRESHOLD = MOVE_THRESHOLD * 8
FOCUS_THRESHOLD = 5
ERROR_PNP_LOC = 1
ERROR_PNP_FALSE = 2
ERROR_MOVE_SORT = 3
MAX_TRIES = 1
add_zone = [(-350,-100),(-200,100)]
OFFSET_X = -5 #-15 #Magic Number
OFFSET_Y = 20 #Magic Number
ERROR_MOVE_SORT = 3
error_file_name = "error_log " + str(datetime.datetime.now()).split('.')[0] + ".txt"

while True:
    print("looping")
    error_log = {}
    error_count = 1
    # spin until you detect a fiducial in the "new card zone"
    new_card, loc, fmap = hiro.wait_for_card(loading_zone=add_zone, focus_threshold=FOCUS_THRESHOLD)
    loc = (loc[0] + OFFSET_X, loc[1] + OFFSET_Y, loc[2])
    # get the notes for the new card and map and POST the new word to the api to update the diagram
    note = fiducial_dict[int(new_card)]
    cur_cards = list(fmap.keys())
    cur_cards.sort()
    cur_notes = [fiducial_dict[int(fid)] for fid in cur_cards]
    cur_locs = [fmap[fid][:2] for fid in cur_cards]

    r = requests.post(Config.API_URL+'/addnote', json={"new_note": note, "notes": cur_notes, "locs": cur_locs, "operations": ["add"]})
    res = r.json()
    # + [0] is for rotation
    new_loc_dict = {notes_dict[res['notes'][i]]:res['locs'][i]+[0] for i in range(len(res['notes']))}
    adds,moves,removes,invalids = hiro.getmoves(new_loc_dict, fmap, add_fid=new_card, dummy_spot=[-200,75,0], threshold = MOVE_THRESHOLD)
    if moves == [-1]:
        relevant_error = {"previous_fmap": fmap, "next_fmap": new_loc_dict}
        error_log[error_count] = {
            "id": ERROR_MOVE_SORT,
            "timestamp": str(datetime.datetime.now()).split('.')[0],
            "extra": relevant_error
        }
        error_count += 1
        moves = []

    print(f"adds:{adds}")
    print(f"moves:{moves}")
    print(f"removes:{removes}")
    print(f"invalids:{invalids}")

    cur_tries = 0
    cur_moves = moves
    invalid_set = set()
    while cur_moves != [] and cur_tries < MAX_TRIES:
        for fid,start,stop in cur_moves:
            hiro.move(np.array([[0],[200],[200]]))
            move_result = hiro.pick_place(start, stop)
            if move_result == False:
                relevant_error = {"start_loc": start, "end_loc": stop}
                error_log[error_count] = {
                    "id": ERROR_PNP_FALSE,
                    "timestamp": str(datetime.datetime.now()).split('.')[0],
                    "extra": relevant_error
                }
                error_count += 1
                invalid_set.add((tuple(start), tuple(stop)))

        check_fiducials = hiro.get_fiducial_map(mask=None)
        _, cur_moves, _, _ = hiro.getmoves(new_loc_dict, check_fiducials, add_fid=new_card, dummy_spot=[-200,75,0], threshold=CHANGE_THRESHOLD)
        if cur_moves == [-1]:
            relevant_error = {"previous_fmap": check_fiducials, "next_fmap": new_loc_dict}
            error_log[error_count] = {
                "id": ERROR_MOVE_SORT,
                "timestamp": str(datetime.datetime.now()).split('.')[0],
                "extra": relevant_error
            }
            error_count += 1
            cur_moves = []
        else:
            cur_moves = [i for i in cur_moves if (tuple(i[1]), tuple(i[2])) not in invalid_set]
            if cur_moves != []:
                if cur_moves != []:
                    relevant_error = {
                        "start_fmap": fmap,
                        "end_fmap": check_fiducials,
                        "desired_fmap": new_loc_dict  
                    }
                    error_log[error_count] = {
                        "id": ERROR_PNP_LOC,
                        "timestamp": str(datetime.datetime.now()).split('.')[0],
                        "extra": relevant_error
                    }
        cur_tries += 1

#    for fid,start,stop in moves:
#        hiro.move(np.array([[0],[200],[200]]))
#        hiro.pick_place(start, stop)
#    assert len(adds) < 2 # assuming we can only add from one location
#    if len(adds) == 1:
#        fid,stop = adds[0]
#        hiro.pick_place(loc,stop)
#    print(r.json())

    if len(adds) == 1:
        cur_tries = 0
        success = False
        fid, stop = adds[0]
        temp_loc = loc
        while not success and cur_tries < MAX_TRIES:
            move_result = hiro.pick_place(temp_loc,stop)
            check_fiducials = hiro.get_fiducial_map(mask=None)
            if fid not in check_fiducials:
                cur_tries = MAX_TRIES
            elif np.linalg.norm(np.array(check_fiducials[fid][:2])-np.array(stop[:2])) > CHANGE_THRESHOLD:
                temp_loc = check_fiducials[fid]
                relevant_error = {
                    "start_fmap": fmap,
                    "end_fmap": check_fiducials,
                    "desired_fmap": new_loc_dict  
                }
                error_log[error_count] = {
                    "id": ERROR_PNP_LOC,
                    "timestamp": str(datetime.datetime.now()).split('.')[0],
                    "extra": relevant_error
                }
                cur_tries += 1
            else:
                success = True

    if error_log != {}:
        with open(error_file_name, "a") as f:
            json.dump(error_log, f)
        

hiro.shutdown()

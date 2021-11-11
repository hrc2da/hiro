from hiro_lightweight_interface import HIROLightweight
from config import Config
import requests
import cv2
import yaml
import numpy as np
import json
import datetime

# instantiate a HIRO object
hiro = HIROLightweight()
add_zone = [(-300,0),(-200,100)]
#add_zone = [(-335, 75), (-260,125)] # gonzalo - more accurate add_zone
MOVE_THRESHOLD = 15
CHANGE_THRESHOLD = MOVE_THRESHOLD * 3
FOCUS_THRESHOLD = 5
ERROR_PNP_LOC = 1
ERROR_PNP_FALSE = 2
ERROR_MOVE_SORT = 3
MAX_TRIES = 2

log_file_name = "general_log " + str(datetime.datetime.now()).split('.')[0] + ".json"
gen_log = {}
iteration = 0
while True:
    gen_log[iteration] = {}
    error_log = {}
    cur_log = {}
    error_count = 0
    print("looping")
    # spin until you detect a fiducial in the "new card zone"
    result = hiro.wait_for_card(loading_zone=add_zone, focus_threshold = FOCUS_THRESHOLD)
    new_card, loc, fmap = result

    cur_log["start"] = {
        "timestamp": str(datetime.datetime.now()).split('.')[0]
        "cur_fmap": fmap,
        "added_card": new_card,
        "card_location": loc
    }
    # get the notes for the new card and map and POST the new word to the api to update the diagram
    note = int(new_card)
    cur_cards = list(fmap.keys())
    cur_cards.sort()
    cur_notes = [int(fid) for fid in cur_cards]
    cur_locs = [fmap[fid][:2] for fid in cur_cards]

    r = requests.post(Config.API_URL+'/addnote', json={"encoding_type":"rgb" ,"new_note": note, "notes": cur_notes, "locs": cur_locs, "operations": ["add"]})
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
    adds,moves,removes,invalids = hiro.getmoves(new_loc_dict, fmap, add_fid=new_card, dummy_spot=[-200,75,0], threshold=MOVE_THRESHOLD)
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

    cur_log["moves"] = {
        "timestamp": str(datetime.datetime.now()).split('.')[0]
        "adds": adds,
        "moves": moves,
        "removes": removes,
        "invalids": invalids
    }
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
        
    assert len(adds) < 2 # assuming we can only add from one location
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
    cur_log["end"] = {
        "timestamp": str(datetime.datetime.now()).split('.')[0],
        "end_fmap": check_fiducials
    }

    cur_log["errors"] = error_log

    gen_log[iteration] = cur_log
    with open(log_file_name, "w") as f:
        json.dump(gen_log, f)

    iteration += 1

    
hiro.shutdown()

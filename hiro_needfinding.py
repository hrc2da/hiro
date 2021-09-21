from hiro_lightweight_interface import HIROLightweight
from config import Config
import requests
import cv2
import yaml
import numpy as np

with open('fiducial_dict.yaml', 'r') as infile:
    fiducial_dict = yaml.load(infile)

notes_dict ={v:k for k,v in fiducial_dict.items()}
# instantiate a HIRO object
hiro = HIROLightweight()
add_zone = [(-300,0),(-200,100)]
while True:
    print("looping")
    # spin until you detect a fiducial in the "new card zone"
    new_card, loc, fmap = hiro.wait_for_card(loading_zone=add_zone)
    # get the notes for the new card and map and POST the new word to the api to update the diagram
    note = fiducial_dict[int(new_card)]
    cur_cards = list(fmap.keys())
    cur_cards.sort()
    cur_notes = [fiducial_dict[int(fid)] for fid in cur_cards]
    cur_locs = [fmap[fid][:2] for fid in cur_cards]

    r = requests.post(Config.API_URL+'/addnote', json={"new_note": note, "notes": cur_notes, "locs": cur_locs, "operations": ["add","split"]})
    res = r.json()
    # + [0] is for rotation
    new_loc_dict = {notes_dict[res['notes'][i]]:res['locs'][i]+[0] for i in range(len(res['notes']))}
    adds,moves,removes,invalids = hiro.getmoves(new_loc_dict, fmap, add_fid=new_card, dummy_spot=[-200,75,0])
    print(f"adds:{adds}")
    print(f"moves:{moves}")
    print(f"removes:{removes}")
    print(f"invalids:{invalids}")
    for fid,start,stop in moves:
        hiro.move(np.array([[0],[200],[200]]))
        hiro.pick_place(start, stop)
    assert len(adds) < 2 # assuming we can only add from one location
    if len(adds) == 1:
        fid,stop = adds[0]
        hiro.pick_place(loc,stop)
    print(r.json())
    

hiro.shutdown()

'''
Just puts cards into clusters blindly
mimicks what actual opperation may look like
tests pick and place fucntionality and cluster configuration
'''

from hiro_interface import *

hiro = HIRO(mute=True)


# test fake clustering

# centerpoints for clusters
c0_centers = [(0, 190), (0, 130), (-80, 130), (80, 130), (-80, 190), (80, 190)]
c1_centers = [(-240, 0), (-240, 60), (-160, 0), (-160, 60), (-320, 0), (-320, 60)] 
c2_centers = [(240, 0), (240, 60), (160, 0), (160, 60), (320, 0), (320, 60)]
c3_centers = [(-260, 230), (-290, 170), (-210, 170), (-180, 230), (-190, 290), (-110, 290)]
c4_centers = [(260, 230), (290, 170), (210, 170), (180, 230), (190, 290), (110, 290)]
cluster_centers = [c0_centers, c1_centers, c2_centers, c3_centers, c4_centers]

seen = [] #ids of notecards already seen

for card_num in range(6): # 6 cards in each cluster
    for cluster in cluster_centers:
        new_id = hiro.find_new_card(seen) # id of new card
        seen.append(new_id)
        print("card {} seen".format(new_id))
        new_loc = hiro.localize_notecard(new_id) #location of new card
        print(new_loc)
        #time.sleep(1) # give user a second to put down card
        hiro.beep(2) # warn that motion is aboout to hapen
        des_loc = cluster[card_num]
        hiro.pick_place(new_loc, des_loc)
        

hiro.move(np.array([[0], [200], [200]]))
print('all done')

#clean up
input("Press Enter to clean up")

pile = (0, 190) # put all cards in first card spot
for card_num in range(6): # 6 cards in each cluster
    for cluster in cluster_centers:
        cluster_loc = cluster[card_num]
        loc = (cluster_loc[0], cluster_loc[1], 0) # assume all cards are horizontal
        hiro.pick_place(loc, pile)

hiro.move(np.array([[0], [200], [200]]))

hiro.disconnect()



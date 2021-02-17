'''
Just puts cards into clusters blindly
mimicks what actual opperation may look like
tests pick and place fucntionality and cluster configuration
'''

from hiro_interface import *
import sys
from copy import copy
from nlp_utils import NoteParser

hiro = HIRO(mute=True)
parser = NoteParser()
temp_photo_path = '/home/pi/hiro/views/view.jpg'



# test fake clustering

# centerpoints for clusters
c0_centers = [(0, 190), (0, 130), (-80, 130), (80, 130), (-80, 190), (80, 190)]
c1_centers = [(-240, 0), (-240, 60), (-160, 0), (-160, 60), (-320, 0), (-320, 60)] 
c2_centers = [(240, 0), (240, 60), (160, 0), (160, 60), (320, 0), (320, 60)]
c3_centers = [(-260, 230), (-290, 170), (-210, 170), (-180, 230), (-190, 290), (-110, 290)]
c4_centers = [(260, 230), (290, 170), (210, 170), (180, 230), (190, 290), (110, 290)]
cluster_capacity = len(c0_centers)
cluster_centers = [c0_centers, c1_centers, c2_centers, c3_centers, c4_centers]
k = len(cluster_centers)
seen = [] #ids of notecards already seen
allwords = dict()
notecards = dict()
word2loc = dict() # dictionary of words to cluster locations tuple (cluster_id, center_id)
current_clusters = [[None for c in range(cluster_capacity)] for i in range(k)]
# for each available spot in the workspace, read a new card and place it in that spot
# for card_num in range(6): # 6 cards in each cluster
#     for cluster in cluster_centers:
#         new_id = hiro.find_new_card(seen) # id of new card
#         print("card {} seen".format(new_id))
#         new_loc = hiro.localize_notecard(new_id) #location of new card
#         #time.sleep(1) # give user a second to put down card
#         hiro.beep(2) # warn that motion is aboout to hapen
#         des_loc = cluster[card_num] # desired locaiton
#         while not hiro.pick_place(new_loc, des_loc):
#             # if card can't be reached keep teeling user to move it
#             print('Please move card closer')
#             time.sleep(5) # give the user some time to move card
#             new_id = hiro.find_new_card(seen) # id of new card
#             new_loc = hiro.localize_notecard(new_id) # new location of new card
#         seen.append(new_id) # add new card id to list of seen cards           

# while true and num_cards < 5*6=30:
# 1. wait for a card
# 2. parse the card to a word vector
# 3. add the word vector the allwords dict
# 4. run clustering/pca on allwords
# 5. reassign cards to locations based on clusters
# 6. for each card, find card and move to new location if necessary
try:
    while len(allwords.items()) < k*cluster_capacity:
        # wait for card (blocking)
        new_id = hiro.find_new_card(seen)
        new_loc = hiro.localize_notecard(new_id)
        # when we get a card, parse it to a word vector
        new_word = parser.photo2txt(temp_photo_path)
        try:
            new_embedding = parser.txt2embedding(new_word)
        except:
            # for now, if we can't parse the word, do nothing and wait for a new card
            print(f"Could not find word {new_word} in vocabulary.")
            continue
        # add the word vector to the allwords dict
        allwords[new_word] = new_embedding
        notecards[new_word] = new_id
        if len(allwords.items())<k:
            # just put the card in the next open cluster
            cluster_id = len(allwords.items())-1
            cluster = cluster_centers[cluster_id]
            des_loc = cluster[0] # desired location is the first spot in the open cluster
            while not hiro.pick_place(new_loc, des_loc):
                # if card can't be reached keep teeling user to move it
                print('Please move card closer')
                time.sleep(5) # give the user some time to move card
                new_id = hiro.find_new_card(seen) # id of new card
                new_loc = hiro.localize_notecard(new_id) # new location of new card
            seen.append(new_id)
            word2loc[new_word] = (cluster_id,0)
            current_clusters[cluster_id][0] = new_word
        else:
            # get new clusters and rearrange cards
            clusters = parser.txt2clusters(list(allwords.keys()),k=k)
            new_clusters = copy(current_clusters)
            print(clusters)
            # assign words in each cluster to a location
            # the dumb way, just go in order
            for i,wordlist in enumerate(clusters):
                for j,word in enumerate(wordlist):
                    new_id = notecards[word]
                    new_loc = hiro.localize_notecard(new_id)
                    des_cluster = i
                    des_cluster_loc = j
                    if new_clusters[i,j] is None:
                        # if the spot is open, then
                        # place card there
                        des_loc = clusters[des_cluster][des_cluster_loc]
                    else:
                        # if the spot is not open,
                        # try to find an open spot
                        des_cluster_loc = None
                        original_des_cluster = des_cluster
                        while des_cluster_loc is None:
                            try:
                                # look for any open spot in the desired cluster
                                des_cluster_loc = new_clusters[des_cluster].index(None)
                            except ValueError:
                                # ok, there's no open spots
                                # for now, we'll try to put it in the cluster next door
                                des_cluster = (des_cluster + 1) % k 
                                if des_cluster == original_des_cluster:
                                    raise ValueError("There are no open spots!")
                        
                        
                    while not hiro.pick_place(new_loc, des_loc):
                        # if card can't be reached keep teeling user to move it
                        print('Please move card closer')
                        time.sleep(5) # give the user some time to move card
                        new_id = hiro.find_new_card(seen) # id of new card
                        new_loc = hiro.localize_notecard(new_id) # new location of new card
                        seen.append(new_id)
                        word2loc[new_word] = (des_cluster,des_cluster_loc)
                        new_clusters[des_cluster][des_cluster_loc] = word


                    
            current_clusters = new_clusters
            # problems with this approach:
            # almost certain to end up placing a card on top of another on recluster
            # very inefficient because we could end up moving all cards in a cluster, e.g. if the cluster id changes
            # alternatives:
            # start with a list of unassigned cluster ids
            # for the first word in the first cluster that is returned, iterate the cluster locations to see if it is in a current cluster.
            # if it is in that cluster, pop that cluster_id from the list of unassigned and assign that cluster id to the first wordlist
            # 

except KeyboardInterrupt:
    sys.exit()

hiro.move(np.array([[0], [200], [200]]))
print('all done')

#clean up
input("Press Enter to clean up")

pile = (0, 190) # put all cards in first card spot
first = True
for card_num in range(6): # 6 cards in each cluster
    for cluster in cluster_centers:
        if not first: # don't have to move first card
            cluster_loc = cluster[card_num]
            loc = (cluster_loc[0], cluster_loc[1], 0) # assume all cards are horizontal
            hiro.pick_place(loc, pile)
        else:
            first = False

hiro.move(np.array([[0], [200], [200]]))

hiro.disconnect()



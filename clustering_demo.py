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
c0_centers = [(-80, 130),  (-80, 190),  (0, 190),    (0, 130),    (80, 130),   (80, 190)]
c1_centers = [(-160, 0),   (-160, 60),  (-240, 60),  (-240, 0),   (-320, 0),   (-320, 60)] 
c2_centers = [(160, 0),    (160, 60),   (240, 60),   (240, 0),    (320, 0),    (320, 60)]
c3_centers = [(-210, 170), (-290, 170), (-260, 230), (-180, 230), (-190, 290), (-110, 290)]
c4_centers = [(210, 170),  (290, 170),  (260, 230),  (180, 230),  (190, 290),  (110, 290)]
cluster_capacity = len(c0_centers)
cluster_centers = [c0_centers, c1_centers, c2_centers] #, c3_centers, c4_centers]
k = len(cluster_centers)
seen = [] #ids of notecards already seen
allwords = dict() # word -> embedding
wordorder = [] # list of words in seen order
cluster_indices = [] # list of cluster assignments in word-seen order
notecards = dict() # word -> fid
word2loc = dict() # dictionary of words to cluster locations tuple (x,y)
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

def get_neighboring_cluster_loc(potential_centers, filled_centers):
	'''
	return a cluster location next to a currently filled spot if possible
	'''
	# find an open spot in that cluster near a filled word if possible
	# search from left to right in cluster centers for a filled spot
	# if the first spot is filled, then search to the right and take the first open spot
	# if not, then, when you find an filled spot, take the one to the left
	# if none of the spots are filled, take the first spot
	# if all of the spots are filled, throw an exception for now (logic to come where we take the best n words)
	found_filled_spot = False
	for i,center in enumerate(potential_centers):
		try:
			occupied = filled_centers.index(center)
			#if no exception, it's filled
			if i==0:
				# break to search right until you get an open spot
				found_filled_spot = True
			elif found_filled_spot == True:
				# this is not the first spot filled and it's all full to the left
				# keep searching or throw an error if we're at the end (all full)
				if i >= len(potential_centers)-1:
					raise(ValueError("Cluster is overfilled!"))
				else:
					continue
			else:
				# this is the first filled spot found
				# take the spot to the left
				return potential_centers[i-1]
		except ValueError:
			#it's empty
			if found_filled_spot == True:
				return center
			else:
				continue
	assert found_filled_spot == False
	return potential_centers[0]

def alignclusters(old_clusters,new_clusters):

	# old_clusters is a list of cluster_ids per word
	# new_clusters is a list of new cluster_ids per word, with the new word added
	# build a frequency table of mappings from old to new
	# 0-0 111
	# 0-1 1
	# 0-2 0
	# 1-0 1
	# 1-1 0
	# 1-2 1111
	# 2-0 1111
	# 2-1 11
	# 2-2 0
	# find the index of the max
	# assign that mapping
	# zero out the others in the same slots
	# repeat

	n_clusters = np.max(old_clusters) + 1
	old_new_clusters = new_clusters[:-1] # get the new clusters for all but the new word
	
	old2newmappings = np.zeros((n_clusters,n_clusters))
	
	for i,new_cluster_id in enumerate(old_new_clusters):
		old2newmappings[old_clusters[i]][new_cluster_id] += 1
	
	# stores what each of the new clusters should be mapped to (so length of num. clusters)
	remappings = np.zeros(n_clusters)
	yet_to_be_mapped_to = list(range(n_clusters))
	yet_to_be_mapped_from = list(range(n_clusters))
	for i in range(n_clusters-1):
		current_max_index = np.unravel_index(old2newmappings.argmax(), old2newmappings.shape)
		old,new = current_max_index
		remappings[new] = old
		# now zero out the row and column 
		old2newmappings[old,:] = 0
		old2newmappings[:,new] = 0
		yet_to_be_mapped_to.remove(old)
		yet_to_be_mapped_from.remove(new)
	remappings[yet_to_be_mapped_from[0]] = yet_to_be_mapped_to[0]
	if len(set(list(remappings))) != len(list(remappings)): # assert that all mappings are unique
		import pdb; pdb.set_trace()
	remapped_clusters = []
	for cid in new_clusters:
		remapped_clusters.append(remappings[cid])
	return remapped_clusters



try:
	while len(allwords.items()) < k*cluster_capacity:
		# wait for card (blocking)
		new_id = hiro.find_new_card(seen,reposition=True)
		new_loc = hiro.localize_notecard(new_id)
		# when we get a card, parse it to a word vector
		try:
			new_word = parser.photo2txt(temp_photo_path)
		except Exception as e:
			print(f"Couldn't find a word, trying again: {e}")
			continue
		try:
			new_embedding = parser.txt2embedding(new_word)
		except:
			# for now, if we can't parse the word, do nothing and wait for a new card
			print(f"Could not find word {new_word} in vocabulary.")
			continue
		# add the word vector to the allwords dict
		allwords[new_word] = new_embedding
		wordorder.append(new_word)
		notecards[new_word] = new_id
		word2loc[new_word] = new_loc[:2]
		print(f"added {new_word} at {word2loc[new_word]}")
		# if we have less than k words, just put it in the next empty cluster
		if len(wordorder)<=k:
			# just put the card in the next open cluster
			cluster_id = len(wordorder)-1
			cluster_indices.append(cluster_id)
			cluster = cluster_centers[cluster_id]
			des_loc = cluster[0] # desired location is the first spot in the open cluster
			while not hiro.pick_place(new_loc, des_loc):
				# if card can't be reached keep teeling user to move it
				print('Please move card closer')
				time.sleep(5) # give the user some time to move card
				new_id = hiro.find_new_card(seen,reposition=True) # id of new card
				new_loc = hiro.localize_notecard(new_id) # new location of new card
			seen.append(new_id)
			word2loc[new_word] = cluster_centers[cluster_id][0]
			current_clusters[cluster_id][0] = new_word
		# otherwise, get clusters
		else:
			# get new clusters and rearrange cards
			# clusters is a list of lists of words, new_cluster_indices is 
			# a vector of len=current # of words assinging cluster ids to each word
			new_cluster_indices, current_clusters = parser.txt2clusters(wordorder,k=k)
			
			# should return a new mapping
			remapped_cluster_indices = alignclusters(cluster_indices, new_cluster_indices)
			
			for word_index, cluster in enumerate(remapped_cluster_indices):
				# if the new cluster assignment is the same as before, skip (excepting the new word)
				if word_index < len(cluster_indices) and cluster == cluster_indices[word_index]:
					continue
				# otherwise, move the card to the new cluster
				else:
					word = wordorder[word_index] # get the word itself
					new_id = notecards[word] # get the fiducial id for that word
					new_loc = (word2loc[word][0], word2loc[word][1], 0) # get the loc of the card from fid
					des_cluster = cluster
					
					potential_centers = cluster_centers[int(des_cluster)]
					filled_centers = list(word2loc.values())
					des_loc = get_neighboring_cluster_loc(potential_centers,filled_centers)
					
				while not hiro.pick_place(new_loc, des_loc):
					# if card can't be reached keep teeling user to move it
					print('Please move card closer')
					time.sleep(5) # give the user some time to move card
					new_id = hiro.find_new_card(seen,reposition=True) # id of new card
					new_loc = hiro.localize_notecard(new_id) # new location of new card
				seen.append(new_id)
				word2loc[word] = des_loc
				# new_clusters[des_cluster][des_cluster_loc] = word

			cluster_indices = remapped_cluster_indices     



			
					




			# new_clusters = copy(current_clusters)
			# print(clusters)
			# # assign words in each cluster to a location
			


			# # the dumb way, just go in order
			# for i,wordlist in enumerate(clusters):
			#     for j,word in enumerate(wordlist):
			#         new_id = notecards[word]
			#         new_loc = (word2loc[word][0], word2loc[word][1], 0)
			#         des_cluster = i
			#         des_cluster_loc = j
			#         if new_clusters[i][j] is None:
			#             # if the spot is open, then
			#             # place card there
			#             des_loc = cluster_centers[des_cluster][des_cluster_loc]
			#         else:
			#             # if the spot is not open,
			#             # try to find an open spot
			#             des_cluster_loc = None
			#             original_des_cluster = des_cluster
			#             while des_cluster_loc is None:
			#                 try:
			#                     # look for any open spot in the desired cluster
			#                     des_cluster_loc = new_clusters[des_cluster].index(None)
			#                 except ValueError:
			#                     # ok, there's no open spots
			#                     # for now, we'll try to put it in the cluster next door
			#                     des_cluster = (des_cluster + 1) % k 
			#                     if des_cluster == original_des_cluster:
			#                         raise ValueError("There are no open spots!")
			#             des_loc = cluster_centers[des_cluster][des_cluster_loc]
			#         print(f'trying to pick {word} from {new_loc} and send to {des_loc}')
			#         while not hiro.pick_place(new_loc, des_loc):
			#             # if card can't be reached keep teeling user to move it
			#             print('Please move card closer')
			#             time.sleep(5) # give the user some time to move card
			#             new_id = hiro.find_new_card(seen,reposition=True) # id of new card
			#             new_loc = hiro.localize_notecard(new_id) # new location of new card
			#         seen.append(new_id)
			#         word2loc[word] = cluster_centers[des_cluster][des_cluster_loc]
			#         new_clusters[des_cluster][des_cluster_loc] = word


					
			# current_clusters = new_clusters
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



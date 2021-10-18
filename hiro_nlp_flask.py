import flask
from flask import request, jsonify, abort
from nlp_utils import NoteParser
import time
import cv2
import string
import numpy as np

from flask import Flask
app = Flask(__name__)


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
			_ = filled_centers.index(center)
			#if no exception, it's filled
			if i==0:
				# break to search right until you get an open spot
				found_filled_spot = True
			elif found_filled_spot == True:
				# this is not the first spot filled and it's all full to the left
				# keep searching or throw an error if we're at the end (all full)
				if i >= len(potential_centers)-1:
					raise(IndexError("Cluster is overfilled!"))
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
		except IndexError:
			# the cluster is full
			# enumerate all the centerpoints
			# sort in order of distance 
			# iterate until you find an empty one
			# actually maybe raise here and handle it outside the function
			return None
	assert found_filled_spot == False
	return potential_centers[0]

def get_isolated_cluster_loc(potential_centers, filled_centers):
	# the goal is to find an open spot in the potential centers that is as isolated as possible
	# do one pass through the potential_clusters, and track the indices of the longest "gap"
	# if the gap is on an end, return the location on the end
	# otherwise, return a location in the middle of the gap if possible
	# if the centers are all filled (no gap), return empty		
	gap_start_index = -1
	longest_gap_start_index = -1
	longest_gap_length = 0
	occupied = [None for i in range(len(potential_centers))]
	for i,c in enumerate(potential_centers):
		try:
			_ = filled_centers.index(c)
			# if filled, set occupied to true
			occupied[i] = True
			if i > 0 and occupied[-1] == False:
				# end the gap, updated maxes
				if i - gap_start_index > longest_gap_length:
					longest_gap_length = i-gap_start_index
					longest_gap_start_index = gap_start_index
			elif i > 0 and occupied[-1] == True:
				# there's no gap, do nothing
				continue
		except ValueError:
			# empty
			# if not in a gap, then start one
			if i == 0:
				gap_start_index = 0
				occupied[0] = False
			elif occupied[-1] == False:
				# we are in a gap, continue it, unless this is the last index
				if i == (len(potential_centers)-1):
					if i - gap_start_index > longest_gap_length:
						longest_gap_length = i-gap_start_index
						longest_gap_start_index = gap_start_index
					elif i-gap_start_index == longest_gap_length and longest_gap_start_index != 0:
						# favor the end points--not sure if this is right or not
						longest_gap_length = i-gap_start_index
						longest_gap_start_index = gap_start_index
				else:
					continue
			elif occupied[-1] == True:
				gap_start_index = i
	
	if longest_gap_length == 0:
		#centers is completely filled
		return None
	elif longest_gap_start_index == 0:
		return potential_centers[0]
	elif longest_gap_start_index + longest_gap_length == len(potential_centers)-1:
		return potential_centers[-1]
	else:
		return potential_centers[int(longest_gap_start_index + longest_gap_length/2)]



def alignclusters(old_clusters,new_clusters,k=None):
	old_clusters = list(map(int,old_clusters))
	new_clusters = list(map(int,new_clusters))
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

	if k is None:
		n_clusters = int(np.max(old_clusters) + 1)
	else:
		n_clusters = k
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
		old2newmappings[old,:] = -1
		old2newmappings[:,new] = -1
		try:
			yet_to_be_mapped_to.remove(old)
		except:
			import pdb; pdb.set_trace()
		yet_to_be_mapped_from.remove(new)
	remappings[yet_to_be_mapped_from[0]] = yet_to_be_mapped_to[0]
	if len(set(list(remappings))) != len(list(remappings)): # assert that all mappings are unique
		import pdb; pdb.set_trace()
	remapped_clusters = []
	for cid in new_clusters:
		remapped_clusters.append(remappings[cid])
	return remapped_clusters




@app.route('/')
def main_page():
	return app.send_static_file('index.html')


@app.route('/sampleembedding', methods=['POST'])
def sample_embedding():
	embeddings = np.array(request.json['embeddings'])
	mean = np.mean(embeddings)
	var = np.std(embeddings)**2
	cov = np.diag(var)
	return jsonify(np.random.multivariate_normal(mean, cov))

@app.route('/txt2embedding', methods=['GET', 'POST'])
def get_text2embedding():
	if request.method == 'POST':
		txt = request.form['word']
		return str(parser.txt2embedding(txt))

@app.route('/txt2pca_km', methods=['GET', 'POST'])
def get_text2pca():
	print("got the request")
	if request.method == 'POST':
		# txt = request.json['word']
		allwords = request.json['allwords']
		try:
			coords = parser.txt2localpca(allwords).tolist()
			clusters = []
			if len(allwords) > 2:
				_, clusters = parser.txt2clusters(allwords)
			return jsonify({"pca":coords, "kmeans":clusters})
		except KeyError as e:
			abort(404,str(e))

@app.route('/photo2txt', methods = ['POST'])
def get_photo2word():    
	if request.method == 'POST':
		f = request.files['image']
		f.save('data/tmp/latest.jpg')
		try:
			word = parser.photo2txt('data/tmp/latest.jpg')
			print(word)
			return word
		except:
			abort(404)

@app.route('/photo2emb', methods= ['POST'])
def get_photo2emb():
	if request.method == 'POST':
		f = request.files['image']
		f.save('data/tmp/latest.jpg')
		img = cv2.imread('data/tmp/latest.jpg')
		cv2.imwrite(f'data/read_imgs/{int(time.time())}.jpg',img) # assuming we don't get lots of requests at once
		try:
			word = parser.photo2txt('data/tmp/latest.jpg').translate(str.maketrans('', '', string.punctuation))
		except Exception as e:
			print(e)
			with open("data/read_imgs/labels.txt", "a") as myfile:
				myfile.write("404\n")
			abort(404) # 404 if cannot parse the photo
		with open("data/read_imgs/labels.txt", "a") as myfile:
			myfile.write(f'{word}\n')
		try:
			embedding = parser.txt2embedding(word).tolist()
			errmsg = None
		except:
			embedding = None
			errmsg = "not in vocab"
		return jsonify({'word':word, 'embedding':embedding, 'err':errmsg})
		

@app.route('/words2clusters', methods=['POST'])
def get_wordclusters():
	if request.method == 'POST':
		new_word = request.json['new_word']
		locs = request.json['location_dict'] # dictionary of word locs
		embeddings = request.json['embedding_dict'] # dictionary of word embeddings
		wordorder = request.json['word_order'] # list of words in order added
		cluster_ids = request.json['cluster_ids'] # parallel list of cluster assignments
		num_clusters = request.json['num_clusters'] # number of clusters
		cluster_centers = request.json['cluster_locs'] # x,y locations for cards in each cluster
		
		# dictionary keyed by word with embeddings and locations
		new_embedding = parser.txt2embedding(new_word)
		wordorder.append(new_word)
		embeddings[new_word] = new_embedding.tolist()

		# before giving the new word a location, do a pass to make sure that
		# every current word is in a valid center (to account for edits)
		flattened_centers = [center for cluster in cluster_centers for center in cluster]
		occupied_locs = list(locs.values())
		for word,loc in locs.items():
			#if the word is not on one of the defined centers (e.g. it moved)
			if loc not in flattened_centers:
				# find a spot for it
				# first, what cluster is it in?
				cur_cluster = cluster_ids[wordorder.index(word)]
				

				potential_centers = cluster_centers[int(cur_cluster)]
				filled_centers = list(locs.values())
				des_loc = get_neighboring_cluster_loc(potential_centers,filled_centers)

				if des_loc == None:
					all_centers = [c for sublist in cluster_centers for c in sublist if c not in filled_centers]
					random_center_idx = np.random.choice(range(len(all_centers)))
					des_loc = all_centers[random_center_idx]
					# I think this should be here???
					# find the cluster the selected center is in
					rand_cluster = 0
					for i,cluster in enumerate(cluster_centers):
						rand_cluster = i
						if des_loc in cluster:
							break
					cluster_ids[wordorder.index(word)] = rand_cluster #random_center_idx // len(cluster_centers[0])
				locs[word] = des_loc







		if len(wordorder) <= num_clusters:
			cluster_id = len(wordorder)-1
			cluster_ids.append(cluster_id)
			cluster = cluster_centers[cluster_id]
			locs[new_word] = cluster[0]
			
		else:
			locs[new_word] = [0,0] #temporary
			# get new clusters and rearrange cards
			# clusters is a list of lists of words, new_cluster_indices is 
			# a vector of len=current # of words assinging cluster ids to each word
			new_cluster_indices, current_clusters = parser.txt2clusters(wordorder,[embeddings[w] for w in wordorder],k=num_clusters)
			
			# should return a new mapping
			remapped_cluster_indices = alignclusters(cluster_ids, new_cluster_indices, k=5)
			
			for word_index, cluster in enumerate(remapped_cluster_indices):
				# if the new cluster assignment is the same as before, skip (excepting the new word)
				if word_index < len(cluster_ids)-1 and cluster == cluster_ids[word_index]:
					continue
				# otherwise, move the card to the new cluster
				else:
					word = wordorder[word_index] # get the word itself
					new_loc = (locs[word][0], locs[word][1]) # get the loc of the card 
					des_cluster = cluster
					
					potential_centers = cluster_centers[int(des_cluster)]
					filled_centers = list(locs.values())
					des_loc = get_neighboring_cluster_loc(potential_centers,filled_centers)
	
					if des_loc == None:
						all_centers = [c for sublist in cluster_centers for c in sublist if c not in filled_centers]
						random_center_idx = np.random.choice(range(len(all_centers)))
						des_loc = all_centers[random_center_idx]
						# I think this should be here???
						# find the cluster the selected center is in
						rand_cluster = 0
						for i,cluster in enumerate(cluster_centers):
							rand_cluster = i
							if des_loc in cluster:
								break
		
						
						remapped_cluster_indices[word_index] = rand_cluster #random_center_idx // len(cluster_centers[0]) ## THIS IS WRONG! ALL CENTERS IS REALLY ALL FILLED CENTERS
					
				locs[word] = des_loc
			cluster_ids = remapped_cluster_indices

		
		return jsonify({
			'location_dict': locs,
			'embedding_dict': embeddings,
			'word_order': wordorder,
			'cluster_ids': [int(c) for c in cluster_ids],
			'num_clusters': num_clusters,
			'cluster_locs': cluster_centers
		})
			

		
		



# @app.route('/photo2embedding', methods = ['POST'])
# def get_photo2embedding():    
#     if request.method == 'POST':
#         f = request.files['the_file']
#         f.save('data/tmp/latest.jpg')

if __name__=='__main__':
	parser = NoteParser()
	app.run()
import os
import numpy as np
import cv2
import json

class DataProvider():
	"this class creates machine-written text for a word list. TODO: change getNext() to return your samples."

	def __init__(self, filedict, crop_type='manual'):
		self.filedict = filedict # keyed on filename, with a list of tuples containing word and position
		self.files = iter(self.filedict.keys())
		self.wordcounts = {k:len(v) for k,v in self.filedict.items()}
		self.num_words = sum(list(self.wordcounts.values()))
		self.cur_file = next(self.files)
		self.cur_file_idx = 0
		self.crop_type = crop_type
		self.idx = 0
		
	def hasNext(self):
		"are there still samples to process?"
		return self.idx < self.num_words

	def getNext(self):
		"TODO: return a sample from your data as a tuple containing the text and the image"
		if self.cur_file_idx >= self.wordcounts[self.cur_file]:
			self.cur_file = next(self.files)
			self.cur_file_idx = 0
		word, top_left, bot_right = self.filedict[self.cur_file][self.cur_file_idx]
		
		img = cv2.imread(self.cur_file)
		if self.crop_type=='manual':
			tlx, tly = top_left
			brx, bry = bot_right
			cropped = img[tly:bry,tlx:brx]
		elif self.crop_type=='rectangle':
			pass
		elif self.crop_type=='segmentationNN':
			pass
		self.cur_file_idx += 1
		self.idx += 1
		return (word, cropped)


def createIAMCompatibleDataset(dataProvider):
	"this function converts the passed dataset to an IAM compatible dataset"

	# create files and directories
	f = open('words.txt', 'w+')
	if not os.path.exists('fsub'):
		os.makedirs('fsub')
	if not os.path.exists('fsub/sub-sub'):
		os.makedirs('fsub/sub-sub')

	# go through data and convert it to IAM format
	ctr = 0
	while dataProvider.hasNext():
		sample = dataProvider.getNext()
		
		# write img
		cv2.imwrite('fsub/sub-sub/sub-sub-%d.png'%ctr, sample[1])
		
		# write filename, dummy-values and text
		line = 'fsub-sub-%d'%ctr + ' X X X X X X X ' + sample[0] + '\n'
		f.write(line)
		
		ctr += 1
		
		
if __name__ == '__main__':
	# with open('/home/dev/scratch/hiro/data_playground/labels/bb2and3and4cropped_single_words.json', 'r') as infile:
	with open('/home/dev/scratch/hiro/data_playground/labels/bb5cropped.json', 'r') as infile:
		fdict = json.load(infile)
	# words = ['some', 'words', 'for', 'which', 'we', 'create', 'text-images']
	dataProvider = DataProvider(fdict)
	createIAMCompatibleDataset(dataProvider)
import cv2
import glob
import json
import sys

labelfile = 'bb5.json'
# run through a directory of images, record bounding boxes on each

#building on https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
boxes = []
annotating = False
image = None
clone = None
tempclone = None
def click_and_annotate(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, image, clone, tempclone
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		annotating = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		annotating = False
		# draw a rectangle around the region of interest
		image = clone.copy()
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		tempclone = image.copy() # get the image without the text
		msg = "Hit s to save or r to reset."
		cv2.putText(image,msg,(50,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
		cv2.imshow("image", image)
	elif event == cv2.EVENT_RBUTTONUP:
		# cancel the annotation
		refPt = []
		annotating = False


imgs = glob.glob('imgs5/imgs/*.jpg')
imgs.sort()

try:
	with open(f'labels/{labelfile}', 'r') as infile:
		annotations = json.load(infile)
except:
	annotations = dict()
if annotations is None:
	annotations = dict()
capFlag = False
for num, img in enumerate(imgs):
	if img in annotations:
		continue
	refPt = []
	boxes = []
	annotating = False
	labeling = False
	label = ''
	# load the image, clone it, and setup the mouse callback function
	image = cv2.imread(img)
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_annotate)
	# keep looping until the 'q' key is pressed
	msg = f"{num}/{len(imgs)}: Click and drag to draw an annotation box."
	cv2.putText(image,msg,(50,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(0) & 0xFF
		if labeling:
			if key == ord('/') or key == 13:
				labeling = False
				boxes.append((label,refPt[0],refPt[1]))
				refPt = []
				oldlabel = label
				label = ''
				image = tempclone.copy()
				clone = image.copy()
				with open('labels/{labelfile}', 'w+') as outfile:
					json.dump(annotations,outfile)
				msg = f"{oldlabel} saved."
				cv2.putText(image,msg,(50,550),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
				msg = f"{num}/{len(imgs)}: Click and drag to draw an annotation box, or n for next."
				cv2.putText(image,msg,(50,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
			elif key == ord('\\'):
				label = ''
				image = tempclone.copy()
				msg = f'Enter label: {label}'
				cv2.putText(image,msg,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
				cv2.putText(image,"Press \\ to clear or / to save.",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
			elif key == 225: #left shift
				capFlag = True
				continue #skip the cap flag reset at the end
			else:
				ch = chr(key)
				if capFlag:
					ch = ch.upper()
				label+=ch
				image = tempclone.copy()
				msg = f'Enter label: {label}'
				cv2.putText(image,msg,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
				cv2.putText(image,"Press \\ to clear or / to save.",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
			capFlag = False
		else:
			# if the 'r' key is pressed, reset the cropping region
			if key == ord("r"):
				image = clone.copy()
				if len(boxes) <1:
					msg = f"{num}/{len(imgs)}: Click and drag to draw an annotation box."
				else:
					msg = f"{num}/{len(imgs)}: Click and drag to draw an annotation box, or n for next."
				cv2.putText(image,msg,(50,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
			# if the 's' key is pressed, save the annotation
			elif key == ord("s"):
				# clone = image.copy()
				image = tempclone.copy()
				msg = f'Enter label: {label}'
				cv2.putText(image,msg,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
				cv2.putText(image,"Press \\ to clear or / to save.",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
				# label = input("Enter the word to annotate: ")
				labeling = True
			elif key == ord("n"):
				annotations[img] = boxes
				break
			elif key == ord("q"):
				cv2.destroyAllWindows()
				with open(f'labels/{labelfile}', 'w+') as outfile:
					json.dump(annotations,outfile)
				sys.exit(0)
				
				
# if there are two reference points, then crop the region of interest
# from teh image and display it
# if len(refPt) == 2:
# 	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# 	cv2.imshow("ROI", roi)
# 	cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
print(annotations)
with open(f'labels/{labelfile}', 'w+') as outfile:
	json.dump(annotations,outfile)
